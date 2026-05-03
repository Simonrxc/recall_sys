import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MovieLensDataset, load_data
from evaluate import build_train_seen_index, load_and_split_data


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
HIDDEN_DIMS = [256, 128, 64]

ABLATION_CONFIGS = [
    {
        "name": "full",
        "user_features": ["user_id", "gender", "age", "occupation", "zip"],
        "item_features": ["movie_id", "genres"],
    },
    {
        "name": "id_only",
        "user_features": ["user_id"],
        "item_features": ["movie_id"],
    },
    {
        "name": "no_user_attrs",
        "user_features": ["user_id"],
        "item_features": ["movie_id", "genres"],
    },
    {
        "name": "no_genres",
        "user_features": ["user_id", "gender", "age", "occupation", "zip"],
        "item_features": ["movie_id"],
    },
    {
        "name": "no_occupation",
        "user_features": ["user_id", "gender", "age", "zip"],
        "item_features": ["movie_id", "genres"],
    },
    {
        "name": "no_zip",
        "user_features": ["user_id", "gender", "age", "occupation"],
        "item_features": ["movie_id", "genres"],
    },
]


def build_mlp(input_dim, hidden_dims):
    layers = []
    for dim in hidden_dims:
        layers.append(nn.Linear(input_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        input_dim = dim
    return nn.Sequential(*layers)


class AblationUserTower(nn.Module):
    def __init__(self, dataset, embed_dim, enabled_features):
        super().__init__()
        self.enabled_features = enabled_features
        self.embeddings = nn.ModuleDict()
        if "user_id" in enabled_features:
            self.embeddings["user_id"] = nn.Embedding(dataset.num_users, embed_dim)
        if "gender" in enabled_features:
            self.embeddings["gender"] = nn.Embedding(dataset.num_genders, embed_dim)
        if "age" in enabled_features:
            self.embeddings["age"] = nn.Embedding(dataset.num_ages, embed_dim)
        if "occupation" in enabled_features:
            self.embeddings["occupation"] = nn.Embedding(dataset.num_occupations, embed_dim)
        if "zip" in enabled_features:
            self.embeddings["zip"] = nn.Embedding(dataset.num_zips, embed_dim)

        input_dim = len(self.embeddings) * embed_dim
        self.mlp = build_mlp(input_dim, HIDDEN_DIMS)

    def forward(self, batch):
        vectors = []
        if "user_id" in self.embeddings:
            vectors.append(self.embeddings["user_id"](batch["user_id"]))
        if "gender" in self.embeddings:
            vectors.append(self.embeddings["gender"](batch["gender"]))
        if "age" in self.embeddings:
            vectors.append(self.embeddings["age"](batch["age"]))
        if "occupation" in self.embeddings:
            vectors.append(self.embeddings["occupation"](batch["occupation"]))
        if "zip" in self.embeddings:
            vectors.append(self.embeddings["zip"](batch["zip"]))

        x = torch.cat(vectors, dim=1)
        return F.normalize(self.mlp(x), p=2, dim=1)


class AblationItemTower(nn.Module):
    def __init__(self, dataset, embed_dim, enabled_features):
        super().__init__()
        self.enabled_features = enabled_features
        self.embeddings = nn.ModuleDict()
        if "movie_id" in enabled_features:
            self.embeddings["movie_id"] = nn.Embedding(dataset.num_movies, embed_dim)
        if "genres" in enabled_features:
            self.embeddings["genres"] = nn.Embedding(
                dataset.num_genres,
                embed_dim,
                padding_idx=0,
            )

        input_dim = len(self.embeddings) * embed_dim
        self.mlp = build_mlp(input_dim, HIDDEN_DIMS)

    def forward(self, movie_ids, genres):
        vectors = []
        if "movie_id" in self.embeddings:
            vectors.append(self.embeddings["movie_id"](movie_ids))
        if "genres" in self.embeddings:
            vectors.append(self.embeddings["genres"](genres).mean(dim=1))

        x = torch.cat(vectors, dim=1)
        return F.normalize(self.mlp(x), p=2, dim=1)


class AblationDSSM(nn.Module):
    def __init__(self, dataset, embed_dim, config):
        super().__init__()
        self.user_tower = AblationUserTower(dataset, embed_dim, config["user_features"])
        self.item_tower = AblationItemTower(dataset, embed_dim, config["item_features"])

    def forward(self, batch, mode):
        user_vec = self.user_tower(batch)
        if mode == "pointwise":
            batch_size, item_count = batch["movie_id"].shape
            item_vec = self.item_tower(
                batch["movie_id"].view(-1),
                batch["genres"].view(-1, 5),
            )
            user_vec = (
                user_vec.unsqueeze(1)
                .expand(-1, item_count, -1)
                .reshape(-1, user_vec.shape[-1])
            )
            return (user_vec * item_vec).sum(dim=1)

        pos_vec = self.item_tower(batch["pos_movie_id"], batch["pos_genres"])
        neg_vec = self.item_tower(batch["neg_movie_id"], batch["neg_genres"])
        return (user_vec * pos_vec).sum(dim=1), (user_vec * neg_vec).sum(dim=1)


def move_batch_to_device(batch, device):
    for key, value in batch.items():
        batch[key] = value.to(device)
    return batch


def train_model(model, train_loader, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() if args.mode == "pointwise" else nn.MarginRankingLoss(margin=args.margin)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, args.device)
            optimizer.zero_grad()

            if args.mode == "pointwise":
                scores = model(batch, mode="pointwise")
                labels = batch["label"].view(-1).to(args.device)
                loss = criterion(scores, labels)
            else:
                pos_score, neg_score = model(batch, mode="pairwise")
                target = torch.ones_like(pos_score)
                loss = criterion(pos_score, neg_score, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(train_loader):.4f}")


def get_item_vectors(model, dataset, device):
    model.eval()
    item_ids = sorted(dataset.movie_features.keys())
    vectors = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[start : start + batch_size]
            movie_ids = torch.LongTensor(batch_ids).to(device)
            genres = torch.LongTensor(
                [dataset.movie_features[mid]["Genres_idx"] for mid in batch_ids]
            ).to(device)
            vectors.append(model.item_tower(movie_ids, genres).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def evaluate_model(model, dataset, train_seen, test_df, args):
    item_vectors = get_item_vectors(model, dataset, args.device)
    total = 0
    recalls = 0.0
    ndcgs = 0.0
    search_k = item_vectors.shape[0]

    test_user_ids = test_df["UserID"].values
    test_target_mids = test_df["MovieID"].values

    for uid_raw, target_mid_raw in tqdm(
        zip(test_user_ids, test_target_mids),
        total=len(test_df),
        desc="Evaluating",
    ):
        try:
            user_idx = dataset.user_encoder.transform([uid_raw])[0]
            target_mid_idx = dataset.movie_encoder.transform([target_mid_raw])[0]
        except ValueError:
            continue

        user_feat = dataset.user_features[user_idx]
        batch = {
            "user_id": torch.LongTensor([user_idx]).to(args.device),
            "gender": torch.LongTensor([user_feat["Gender_idx"]]).to(args.device),
            "age": torch.LongTensor([user_feat["Age_idx"]]).to(args.device),
            "occupation": torch.LongTensor([user_feat["Occupation_idx"]]).to(args.device),
            "zip": torch.LongTensor([user_feat["Zip_idx"]]).to(args.device),
        }

        with torch.no_grad():
            user_vec = model.user_tower(batch).cpu().numpy()

        scores = np.dot(user_vec, item_vectors.T)[0]
        indices = np.argpartition(scores, -search_k)[-search_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        seen_items = train_seen.get(user_idx, set())
        rec_list = np.array([mid for mid in indices if mid not in seen_items])
        top_k_recs = rec_list[:50]

        total += 1
        if target_mid_idx in top_k_recs:
            recalls += 1.0
            rank = np.where(top_k_recs == target_mid_idx)[0][0]
            ndcgs += 1.0 / math.log2(rank + 2)

    return {
        "Recall@50": recalls / total if total else 0.0,
        "NDCG@50": ndcgs / total if total else 0.0,
        "total_users": total,
    }


def save_results(result_dir, records, best_record, args):
    csv_path = result_dir / "feature_ablation_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
        fieldnames = [
            "name",
            "mode",
            "epochs",
            "embed_dim",
            "lr",
            "batch_size",
            "user_features",
            "item_features",
            "Recall@50",
            "NDCG@50",
            "total_users",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    json_path = result_dir / "feature_ablation_results.json"
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "best_by_recall_at_50": best_record,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    return csv_path, json_path


def build_parser():
    parser = argparse.ArgumentParser(description="Run DSSM feature ablation experiments.")
    parser.add_argument("--mode", type=str, default="pairwise", choices=["pointwise", "pairwise"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--neg_ratio", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    return parser


def main():
    args = build_parser().parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.output_dir).resolve() / f"feature_ablation_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=True)

    users, movies, train_df, test_df = load_and_split_data()
    train_dataset = MovieLensDataset(
        train_df,
        users,
        movies,
        mode=args.mode,
        neg_ratio=args.neg_ratio,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    train_seen = build_train_seen_index(train_df, train_dataset)

    records = []
    best_record = None
    for config in ABLATION_CONFIGS:
        print("=" * 80)
        print(f"Running feature ablation: {config['name']}")
        print("=" * 80)

        model = AblationDSSM(train_dataset, args.embed_dim, config).to(args.device)
        train_model(model, train_loader, args)
        metrics = evaluate_model(model, train_dataset, train_seen, test_df, args)

        record = {
            "name": config["name"],
            "mode": args.mode,
            "epochs": args.epochs,
            "embed_dim": args.embed_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "user_features": "|".join(config["user_features"]),
            "item_features": "|".join(config["item_features"]),
            **metrics,
        }
        records.append(record)
        if best_record is None or record["Recall@50"] > best_record["Recall@50"]:
            best_record = record

        csv_path, json_path = save_results(result_dir, records, best_record, args)
        print(
            f"{config['name']}: "
            f"Recall@50={record['Recall@50']:.4f}, "
            f"NDCG@50={record['NDCG@50']:.4f}"
        )

    print("=" * 80)
    print("Feature ablation finished.")
    print(f"Best by Recall@50: {best_record['name']} ({best_record['Recall@50']:.4f})")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
