import argparse
import json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from dataset import MovieLensDataset, load_data
from model import DSSM
import math

try:
    import faiss
except ImportError:
    faiss = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def get_dataset_info(data_dir):
    """读取转换目录元信息，提取 ml-1m/ml-20m 等数据集名称。"""
    metadata_path = os.path.join(data_dir, "metadata.json") if data_dir else None
    source_dataset = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            source_dataset = json.load(f).get("source_dataset")

    dataset_name = os.path.basename(os.path.normpath(source_dataset or data_dir or "unknown"))
    return {
        "dataset_name": dataset_name,
        "source_dataset": source_dataset or data_dir,
        "data_dir": data_dir,
    }


def save_experiment_results(metrics, config, output_dir=OUTPUT_DIR):
    """保存评估结果到 output 目录；该目录已在 .gitignore 中忽略。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": "dssm",
        "timestamp": timestamp,
        "config": config,
        "metrics": metrics,
    }

    json_path = os.path.join(output_dir, f"experiment_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    rows = []
    for k, values in metrics["by_k"].items():
        rows.append(
            {
                "timestamp": timestamp,
                "model": "dssm",
                "k": k,
                "hr": values["hr"],
                "ndcg": values["ndcg"],
                "total_users": metrics["total_users"],
                "dataset_name": config["dataset_name"],
                "source_dataset": config.get("source_dataset"),
                "data_dir": config["data_dir"],
                "model_path": config["model_path"],
                "embed_dim": config["embed_dim"],
                "device": config["device"],
                "retrieval_backend": config["retrieval_backend"],
                "num_users": config["num_users"],
                "num_movies": config["num_movies"],
                "num_ratings": config["num_ratings"],
            }
        )

    csv_path = os.path.join(output_dir, "experiment_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Experiment results saved to {json_path} and {csv_path}")

def load_and_split_data():
    """
    加载数据并按 Leave-One-Out 划分测试集
    """
    print("Loading data...")
    users, movies, ratings = load_data()
    
    # 按用户和时间排序
    ratings = ratings.sort_values(by=['UserID', 'Timestamp'])
    
    test_data = []
    
    # 简单的 Leave-One-Out
    # 为了保证 ID 映射一致，我们需要先用全量数据初始化 Dataset
    # 这样 Dataset 内部的 Encoder 就会包含所有 User 和 Movie
    # 然后我们在评估时，只取测试集的那部分交互
    
    # 按用户分组，取最后一条作为测试
    last_interactions = ratings.groupby('UserID').tail(1)
    
    # 构造测试集 DataFrame: UserID, MovieID
    test_df = last_interactions[['UserID', 'MovieID']].copy()
    
    return users, movies, ratings, test_df

def get_embeddings(model, dataset, device):
    """
    生成所有 User 和 Item 的 Embedding
    """
    model.eval()
    
    # 1. 生成所有 Movie 的 Embedding
    print("Generating Item Embeddings...")
    item_embeddings = []
    item_ids = []
    
    # 遍历所有 MovieID (通过 dataset.movie_features)
    # dataset.movie_features key 是 MovieID_idx
    # 我们需要按顺序生成，或者记录 ID
    
    # 构造 Batch
    movie_indices = list(dataset.movie_features.keys())
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(movie_indices), batch_size):
            batch_mids = movie_indices[i : i + batch_size]
            
            # 准备输入
            mids_tensor = torch.LongTensor(batch_mids).to(device)
            
            genres_list = [dataset.movie_features[m]['Genres_idx'] for m in batch_mids]
            genres_tensor = torch.LongTensor(genres_list).to(device)
            
            # Forward Item Tower
            emb = model.item_tower(mids_tensor, genres_tensor)
            item_embeddings.append(emb.cpu().numpy())
            item_ids.extend(batch_mids)
            
    item_embeddings = np.concatenate(item_embeddings, axis=0)
    # 建立 MovieID_idx -> Embedding 索引 (其实就是数组下标，如果我们按顺序的话)
    # 这里 item_ids 和 item_embeddings 是对应的
    
    # 为了方便检索，我们需要构建一个 map: MovieID_idx -> Vector
    # 或者直接用 item_embeddings 作为库，下标即 ID (前提是 ID 是 0..N-1 连续的)
    # LabelEncoder 产生的 ID 是连续的 0..N-1
    # 只要我们按 0..N-1 的顺序生成即可
    
    # 重新按 0..N-1 排序
    sorted_indices = np.argsort(item_ids)
    final_item_embeddings = item_embeddings[sorted_indices]
    
    return final_item_embeddings

def evaluate(args):
    # 1. 加载数据
    users, movies, ratings, test_df = load_and_split_data()
    
    # 2. 初始化 Dataset (用于获取 Encoder 和特征)
    # 注意：这里会重新 fit encoder，只要数据源没变，结果一致
    dataset = MovieLensDataset(ratings, users, movies, mode='pointwise')
    
    # 3. 加载模型
    print(f"Loading model from {args.model_path}...")
    model = DSSM(
        num_users=dataset.num_users,
        num_genders=dataset.num_genders,
        num_ages=dataset.num_ages,
        num_occupations=dataset.num_occupations,
        num_zips=dataset.num_zips,
        num_movies=dataset.num_movies,
        num_genres=dataset.num_genres,
        embed_dim=args.embed_dim
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    
    # 4. 生成物品向量库
    item_vectors = get_embeddings(model, dataset, args.device)
    print(f"Item Vectors Shape: {item_vectors.shape}")
    
    # 5. 构建 FAISS 索引
    if faiss:
        print("Building FAISS index...")
        # Inner Product for Cosine Similarity (vectors are normalized)
        index = faiss.IndexFlatIP(item_vectors.shape[1])
        index.add(item_vectors)
    else:
        print("FAISS not found, using numpy dot product.")
        
    # 6. 评估
    print("Evaluating...")
    ks = [3, 5, 10]
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    total = 0
    
    # 遍历测试集用户
    # 批量处理 User
    test_user_ids = test_df['UserID'].values
    test_target_mids = test_df['MovieID'].values
    
    batch_size = 256
    
    for i in tqdm(range(0, len(test_user_ids), batch_size)):
        batch_uids_raw = test_user_ids[i : i+batch_size]
        batch_target_mids_raw = test_target_mids[i : i+batch_size]
        
        # 转换 ID
        # 注意处理 Unknown ID (虽然 Leave-One-Out 不应该有 Unknown，但为了健壮性)
        batch_uids_idx = []
        valid_indices = [] # 记录有效的 batch index
        
        for idx, uid in enumerate(batch_uids_raw):
            # LabelEncoder transform
            try:
                # 这种方式比较慢，但为了简单先这样
                # 实际应该用 dataset.user_encoder.transform，但要注意处理 unseen
                # 这里直接查 dataset.user_features 的 key 应该不行，key 是 idx
                # 我们需要原始 ID 到 idx 的映射
                # dataset.user_encoder 是 LabelEncoder
                u_idx = dataset.user_encoder.transform([uid])[0]
                batch_uids_idx.append(u_idx)
                valid_indices.append(idx)
            except ValueError:
                continue
                
        if not batch_uids_idx:
            continue
            
        # 准备 User 特征输入
        user_feats = [dataset.user_features[u_idx] for u_idx in batch_uids_idx]
        
        u_gender = torch.LongTensor([f['Gender_idx'] for f in user_feats]).to(args.device)
        u_age = torch.LongTensor([f['Age_idx'] for f in user_feats]).to(args.device)
        u_occ = torch.LongTensor([f['Occupation_idx'] for f in user_feats]).to(args.device)
        u_zip = torch.LongTensor([f['Zip_idx'] for f in user_feats]).to(args.device)
        u_id_tensor = torch.LongTensor(batch_uids_idx).to(args.device)
        
        # 生成 User Vector
        with torch.no_grad():
            user_vecs = model.user_tower(u_id_tensor, u_gender, u_age, u_occ, u_zip).cpu().numpy()
            
        # 检索 Top-K (最大 K)
        max_k = max(ks)
        
        if faiss:
            D, I = index.search(user_vecs, max_k)
        else:
            # Numpy dot product
            scores = np.dot(user_vecs, item_vectors.T) # (B, N_items)
            # Top-K
            # argpartition + sort is faster than argsort
            I = []
            for j in range(len(scores)):
                ind = np.argpartition(scores[j], -max_k)[-max_k:]
                ind = ind[np.argsort(scores[j][ind])[::-1]]
                I.append(ind)
            I = np.array(I)
            
        # 计算指标
        for j, valid_idx in enumerate(valid_indices):
            target_mid_raw = batch_target_mids_raw[valid_idx]
            try:
                target_mid_idx = dataset.movie_encoder.transform([target_mid_raw])[0]
            except ValueError:
                continue
                
            rec_list = I[j] # List of MovieID_idx
            
            total += 1
            for k in ks:
                top_k_recs = rec_list[:k]
                if target_mid_idx in top_k_recs:
                    hits[k] += 1
                    rank = np.where(top_k_recs == target_mid_idx)[0][0]
                    ndcgs[k] += 1.0 / math.log2(rank + 2)
                    
    print("-" * 40)
    print("DSSM Evaluation Results:")
    print("-" * 40)
    by_k = {}
    for k in ks:
        hr = hits[k] / total
        ndcg = ndcgs[k] / total
        by_k[k] = {"hr": hr, "ndcg": ndcg}
        print(f"HR@{k}: {hr:.4f}")
        print(f"NDCG@{k}: {ndcg:.4f}")
    print("-" * 40)

    metrics = {
        "total_users": total,
        "by_k": by_k,
    }
    data_dir = getattr(__import__("dataset"), "DATA_DIR", None)
    config = {
        **get_dataset_info(data_dir),
        "model_path": args.model_path,
        "embed_dim": args.embed_dim,
        "device": args.device,
        "retrieval_backend": "faiss" if faiss else "numpy",
        "num_users": len(users),
        "num_movies": len(movies),
        "num_ratings": len(ratings),
        "num_test_users": len(test_df),
    }
    save_experiment_results(metrics, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='dssm_pointwise.pth')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    evaluate(args)

