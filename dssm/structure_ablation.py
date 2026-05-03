import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MovieLensDataset
from evaluate import build_train_seen_index, load_and_split_data


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

ABLATION_CONFIGS = [
    {"name": "mlp_1_layer", "hidden_dims": [64], "description": "shallow"},
    {"name": "mlp_2_layers", "hidden_dims": [128, 64], "description": "medium"},
    {"name": "mlp_3_layers", "hidden_dims": [256, 128, 64], "description": "baseline"},
    {"name": "mlp_4_layers", "hidden_dims": [512, 256, 128, 64], "description": "deeper"},
]


def build_mlp(input_dim, hidden_dims):
    layers = []
    for dim in hidden_dims:
        layers.append(nn.Linear(input_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        input_dim = dim
    return nn.Sequential(*layers)


class StructureUserTower(nn.Module):
    def __init__(self, dataset, embed_dim, hidden_dims):
        super().__init__()
        self.user_emb = nn.Embedding(dataset.num_users, embed_dim)
        self.gender_emb = nn.Embedding(dataset.num_genders, embed_dim)
        self.age_emb = nn.Embedding(dataset.num_ages, embed_dim)
        self.occ_emb = nn.Embedding(dataset.num_occupations, embed_dim)
        self.zip_emb = nn.Embedding(dataset.num_zips, embed_dim)
        self.mlp = build_mlp(5 * embed_dim, hidden_dims)

    def forward(self, batch):
        x = torch.cat(
            [
                self.user_emb(batch["user_id"]),
                self.gender_emb(batch["gender"]),
                self.age_emb(batch["age"]),
                self.occ_emb(batch["occupation"]),
                self.zip_emb(batch["zip"]),
            ],
            dim=1,
        )
        return F.normalize(self.mlp(x), p=2, dim=1)


class StructureItemTower(nn.Module):
    def __init__(self, dataset, embed_dim, hidden_dims):
        super().__init__()
        self.movie_emb = nn.Embedding(dataset.num_movies, embed_dim)
        self.genre_emb = nn.Embedding(dataset.num_genres, embed_dim, padding_idx=0)
        self.mlp = build_mlp(2 * embed_dim, hidden_dims)

    def forward(self, movie_ids, genres):
        movie_vec = self.movie_emb(movie_ids)
        genre_vec = self.genre_emb(genres).mean(dim=1)
        x = torch.cat([movie_vec, genre_vec], dim=1)
        return F.normalize(self.mlp(x), p=2, dim=1)


class StructureDSSM(nn.Module):
    def __init__(self, dataset, embed_dim, hidden_dims):
        super().__init__()
        self.user_tower = StructureUserTower(dataset, embed_dim, hidden_dims)
        self.item_tower = StructureItemTower(dataset, embed_dim, hidden_dims)

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

    for uid_raw, target_mid_raw in tqdm(
        test_df[["UserID", "MovieID"]].itertuples(index=False),
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
    csv_path = result_dir / "structure_ablation_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
        fieldnames = [
            "name",
            "description",
            "hidden_dims",
            "mode",
            "epochs",
            "embed_dim",
            "lr",
            "batch_size",
            "Recall@50",
            "NDCG@50",
            "total_users",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    json_path = result_dir / "structure_ablation_results.json"
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
    parser = argparse.ArgumentParser(description="Run DSSM structure ablation experiments.")
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
    result_dir = Path(args.output_dir).resolve() / f"structure_ablation_{run_id}"
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
        print(f"Running structure ablation: {config['name']}")
        print("=" * 80)

        model = StructureDSSM(train_dataset, args.embed_dim, config["hidden_dims"]).to(args.device)
        train_model(model, train_loader, args)
        metrics = evaluate_model(model, train_dataset, train_seen, test_df, args)

        record = {
            "name": config["name"],
            "description": config["description"],
            "hidden_dims": "|".join(map(str, config["hidden_dims"])),
            "mode": args.mode,
            "epochs": args.epochs,
            "embed_dim": args.embed_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
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
    print("Structure ablation finished.")
    print(f"Best by Recall@50: {best_record['name']} ({best_record['Recall@50']:.4f})")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
