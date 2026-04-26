#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import math
from gensim.models import Word2Vec
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIRS = [
    os.path.join(REPO_ROOT, "convert_dataset"),
    os.path.join(REPO_ROOT, "converted_dataset"),
]
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def resolve_data_dir():
    """定位 convert_dataset.py 生成的统一数据目录。"""
    env_data_dir = os.environ.get("ITEMCF_DATA_DIR") or os.environ.get("RECALL_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(env_data_dir)

    for data_dir in DEFAULT_DATA_DIRS:
        if os.path.isdir(data_dir):
            return data_dir

    return DEFAULT_DATA_DIRS[0]


DATA_DIR = resolve_data_dir()


def get_dataset_info():
    """读取转换元信息，标识本次实验使用的是 ml-1m、ml-20m 等哪个数据集。"""
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    source_dataset = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            source_dataset = json.load(f).get("source_dataset")

    dataset_name = os.path.basename(os.path.normpath(source_dataset or DATA_DIR))
    return {
        "dataset_name": dataset_name,
        "source_dataset": source_dataset or DATA_DIR,
    }


def load_converted_data():
    """加载转换后的 movies.csv 和 ratings.csv，并映射为算法内部字段名。"""
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    missing_files = [path for path in [movies_path, ratings_path] if not os.path.exists(path)]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"未找到转换后的数据文件: {missing}\n"
            "请先运行: python convert_dataset.py -o convert_dataset"
        )

    movies = pd.read_csv(movies_path).rename(
        columns={
            "movie_id": "MovieID",
            "title": "Title",
            "genres": "Genres",
        }
    )
    ratings = pd.read_csv(ratings_path).rename(
        columns={
            "user_id": "UserID",
            "movie_id": "MovieID",
            "rating": "Rating",
            "timestamp": "Timestamp",
        }
    )

    movies = movies[["MovieID", "Title", "Genres"]].copy()
    ratings = ratings[["UserID", "MovieID", "Rating", "Timestamp"]].copy()
    movies["Genres"] = movies["Genres"].fillna("(no genres listed)").astype(str)
    return movies, ratings

def load_and_split_data():
    """
    加载数据并按 Leave-One-Out 方式划分为训练集和测试集
    """
    print(f"Loading converted data from {DATA_DIR} and splitting data...")
    movies, ratings = load_converted_data()
    
    # 按用户和时间排序
    ratings = ratings.sort_values(by=['UserID', 'Timestamp'])
    
    train_data = []
    test_data = []
    
    # 按用户分组划分
    for user_id, group in tqdm(ratings.groupby('UserID'), desc="Splitting data"):
        # 转换为列表: [(MovieID, Rating, Timestamp), ...]
        user_history = group[['MovieID', 'Rating', 'Timestamp']].values.tolist()
        
        if len(user_history) < 2:
            # 如果交互少于2个，全部放入训练集（无法做测试）
            train_data.extend([[user_id, *item] for item in user_history])
            continue
            
        # 最后一个作为测试集
        test_item = user_history[-1]
        test_data.append([user_id, *test_item])
        
        # 其余作为训练集
        train_items = user_history[:-1]
        train_data.extend([[user_id, *item] for item in train_items])
        
    # 转换为 DataFrame
    train_df = pd.DataFrame(train_data, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    test_df = pd.DataFrame(test_data, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    return movies, train_df, test_df

def train_genre_embeddings(movies, vector_size=32, window=5, min_count=1):
    """训练 Genre Word2vec (复用之前的逻辑)"""
    print("Training Genre Word2vec...")
    sentences = [genres.split('|') for genres in movies['Genres']]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, seed=42)
    genre_vectors = {genre: model.wv[genre] for genre in model.wv.index_to_key}
    return genre_vectors

def compute_movie_embeddings(movies, genre_vectors, vector_size=32):
    """计算电影 Embedding (复用之前的逻辑)"""
    print("Computing Movie Embeddings...")
    movie_embeddings = {}
    for _, row in movies.iterrows():
        movie_id = row['MovieID']
        genres = row['Genres'].split('|')
        vectors = [genre_vectors[g] for g in genres if g in genre_vectors]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            norm = np.linalg.norm(avg_vector)
            if norm > 0:
                avg_vector = avg_vector / norm
            movie_embeddings[movie_id] = avg_vector
        else:
            movie_embeddings[movie_id] = np.zeros(vector_size)
    return movie_embeddings

def build_item_sim_index(movie_embeddings, top_k=50):
    """建立 Item-Item 相似度索引"""
    print("Building Item-Item Similarity Index...")
    movie_ids = list(movie_embeddings.keys())
    # 建立 ID 映射以便使用矩阵运算
    id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    
    embedding_matrix = np.array([movie_embeddings[mid] for mid in movie_ids])
    
    # 计算全量相似度矩阵
    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    
    item_sim_index = {}
    
    for i, mid in enumerate(tqdm(movie_ids, desc="Indexing items")):
        scores = similarity_matrix[i]
        # 获取 top_k 相似 (排除自己)
        # argsort 是升序，取最后 k+1 个，然后反转
        top_indices = np.argsort(scores)[-(top_k+1):][::-1]
        
        similar_items = []
        for idx in top_indices:
            sim_mid = movie_ids[idx]
            if sim_mid != mid:
                similar_items.append((sim_mid, float(scores[idx])))
        
        item_sim_index[mid] = similar_items[:top_k]
        
    return item_sim_index

def build_user_history_index(train_df):
    """建立用户历史索引 (只基于训练集)"""
    print("Building User History Index from Train Set...")
    # 按时间倒序排序，方便取 last-n
    train_df = train_df.sort_values(by=['UserID', 'Timestamp'], ascending=[True, False])
    
    user_history_index = defaultdict(list)
    
    # 转换为 numpy 迭代更快
    data = train_df[['UserID', 'MovieID', 'Rating']].values
    
    for row in data:
        uid, mid, rating = int(row[0]), int(row[1]), float(row[2])
        user_history_index[uid].append((mid, rating))
        
    return user_history_index

def recommend(user_id, user_history_index, item_sim_index, last_n=20, top_k_sim=20, top_n_rec=100):
    """
    核心推荐逻辑
    1. 获取用户 last-n 历史
    2. 对每个历史物品找 top-k 相似物品
    3. 加权打分
    4. 返回 top-n 推荐
    """
    if user_id not in user_history_index:
        return []
    
    # 1. 获取用户近期感兴趣的物品列表 (last-n)
    history = user_history_index[user_id][:last_n]
    
    candidates = defaultdict(float)
    watched_items = set(mid for mid, _ in user_history_index[user_id])
    
    # 2. & 3. 遍历历史物品，找相似物品并打分
    for hist_mid, hist_rating in history:
        if hist_mid in item_sim_index:
            # 取出 top-k 相似物品
            similar_items = item_sim_index[hist_mid][:top_k_sim]
            
            for sim_mid, sim_score in similar_items:
                # 过滤掉用户已经看过的物品
                if sim_mid in watched_items:
                    continue
                
                # 4. 预估兴趣分数: 相似度 * 历史评分 (这里简单累加)
                candidates[sim_mid] += sim_score * hist_rating
                
    # 5. 返回分数最高的 top-n
    recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n_rec]
    return [mid for mid, score in recs]


def save_experiment_results(metrics, metadata):
    """保存实验指标到 output 目录；该目录已在 .gitignore 中忽略。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "module": "itemcf",
        "timestamp": timestamp,
        "data_dir": DATA_DIR,
        "metadata": metadata,
        "metrics": metrics,
    }

    json_path = os.path.join(OUTPUT_DIR, f"experiment_itemcf_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, f"experiment_itemcf_{timestamp}.csv")
    row = {
        "timestamp": timestamp,
        "model": "itemcf",
        "dataset_name": metadata["dataset_name"],
        "source_dataset": metadata["source_dataset"],
        "data_dir": DATA_DIR,
        "train_samples": metadata["train_samples"],
        "test_samples": metadata["test_samples"],
        "num_movies": metadata["num_movies"],
    }
    for k, values in metrics.items():
        k_label = str(k).lstrip("@")
        row[f"Recall@{k_label}"] = values["recall"]
        row[f"HR@{k_label}"] = values["hr"]
        row[f"NDCG@{k_label}"] = values["ndcg"]

    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Experiment results saved to {json_path} and {csv_path}")

def calculate_metrics(test_df, user_history_index, item_sim_index, ks=[50, 100, 200]):
    """计算 Recall@K、HR@K 和 NDCG@K。Leave-One-Out 下 Recall@K 等价于 HR@K。"""
    print("\nCalculating metrics...")
    
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    total_users = 0
    
    # 遍历测试集
    test_data = test_df[['UserID', 'MovieID']].values
    
    for uid, target_mid in tqdm(test_data, desc="Evaluating"):
        total_users += 1
        
        # 获取推荐列表 (Top 100 足够涵盖最大的 K)
        max_k = max(ks)
        rec_list = recommend(uid, user_history_index, item_sim_index, last_n=50, top_k_sim=20, top_n_rec=max_k)
        
        # 计算指标
        for k in ks:
            # 截取 Top K
            top_k_recs = rec_list[:k]
            
            if target_mid in top_k_recs:
                # Hit Ratio
                hits[k] += 1
                
                # NDCG
                rank = top_k_recs.index(target_mid)
                ndcgs[k] += 1.0 / math.log2(rank + 2)
                
    # 输出结果
    print("-" * 40)
    print("Evaluation Results:")
    print("-" * 40)
    metrics = {}
    for k in ks:
        hr = hits[k] / total_users if total_users else 0.0
        recall = hr
        ndcg = ndcgs[k] / total_users if total_users else 0.0
        metrics[f"@{k}"] = {"recall": recall, "hr": hr, "ndcg": ndcg}
        print(f"Recall@{k}: {recall:.4f}")
        print(f"HR@{k}: {hr:.4f}")
        print(f"NDCG@{k}: {ndcg:.4f}")
    print("-" * 40)
    return metrics

def main():
    # 1. 加载并划分数据
    movies, train_df, test_df = load_and_split_data()
    
    # 2. 训练 Genre Embeddings (使用全量 Movies 数据是允许的，因为是 Content Feature)
    genre_vectors = train_genre_embeddings(movies)
    
    # 3. 计算 Movie Embeddings
    movie_embeddings = compute_movie_embeddings(movies, genre_vectors)
    
    # 4. 建立 Item-Item 索引 (基于 Movie Embeddings 计算相似度)
    item_sim_index = build_item_sim_index(movie_embeddings)
    
    # 5. 建立 User History 索引 (只使用训练集!)
    user_history_index = build_user_history_index(train_df)
    
    # 6. 评估
    metrics = calculate_metrics(test_df, user_history_index, item_sim_index, ks=[50, 100, 200])
    save_experiment_results(
        metrics,
        {
            **get_dataset_info(),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "num_movies": len(movies),
        },
    )

if __name__ == "__main__":
    main()








