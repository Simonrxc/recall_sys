#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIRS = [
    os.path.join(REPO_ROOT, "convert_dataset"),
    os.path.join(REPO_ROOT, "converted_dataset"),
]


def resolve_data_dir():
    """定位 convert_dataset.py 生成的统一数据目录。"""
    env_data_dir = os.environ.get("USERCF_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(env_data_dir)

    for data_dir in DEFAULT_DATA_DIRS:
        if os.path.isdir(data_dir):
            return data_dir

    return DEFAULT_DATA_DIRS[0]


DATA_DIR = resolve_data_dir()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def get_dataset_info():
    """读取转换目录元信息，识别当前实验使用的 MovieLens 数据集版本。"""
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

def load_and_split_data():
    """
    加载数据并按 Leave-One-Out 方式划分为训练集和测试集
    """
    print(f"Loading converted data from {DATA_DIR}...")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"未找到转换后的数据文件: {ratings_path}\n"
            "请先运行: python convert_dataset.py -o convert_dataset"
        )

    ratings = pd.read_csv(ratings_path).rename(
        columns={
            "user_id": "UserID",
            "movie_id": "MovieID",
            "rating": "Rating",
            "timestamp": "Timestamp",
        }
    )
    ratings = ratings[["UserID", "MovieID", "Rating", "Timestamp"]].copy()
    
    # 按用户和时间排序
    ratings = ratings.sort_values(by=['UserID', 'Timestamp'])
    
    train_data = []
    test_data = []
    
    # 按用户分组划分
    for user_id, group in tqdm(ratings.groupby('UserID'), desc="Splitting data"):
        # 转换为列表: [(MovieID, Rating, Timestamp), ...]
        user_history = group[['MovieID', 'Rating', 'Timestamp']].values.tolist()
        
        if len(user_history) < 2:
            train_data.extend([[user_id, *item] for item in user_history])
            continue
            
        # 最后一个作为测试集
        test_item = user_history[-1]
        test_data.append([user_id, *test_item])
        
        # 其余作为训练集
        train_items = user_history[:-1]
        train_data.extend([[user_id, *item] for item in train_items])
        
    train_df = pd.DataFrame(train_data, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    test_df = pd.DataFrame(test_data, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    return train_df, test_df

def build_indices(train_df, sim_top_k=50):
    """
    建立 UserCF 所需的索引
    """
    print("Building Indices...")
    
    # 1. User-Item Index (User的历史)
    # 按时间倒序排序，方便取 Last-n
    train_df_sorted = train_df.sort_values(by=['UserID', 'Timestamp'], ascending=[True, False])
    
    user_item_index = defaultdict(list)     # {uid: [mid, ...]} 用于计算交集 (set更快，但这里需要顺序取last-n，所以分开存)
    user_item_set = defaultdict(set)        # {uid: {mid}} 用于快速判断是否存在
    user_item_history = defaultdict(list)   # {uid: [(mid, rating), ...]} 完整历史
    
    data = train_df_sorted[['UserID', 'MovieID', 'Rating']].values
    for row in data:
        uid, mid, rating = int(row[0]), int(row[1]), float(row[2])
        user_item_index[uid].append(mid)
        user_item_set[uid].add(mid)
        user_item_history[uid].append((mid, rating))
        
    # 2. Item-User Index (Inverted Index)
    item_user_index = defaultdict(set)
    for uid, mids in user_item_set.items():
        for mid in mids:
            item_user_index[mid].add(uid)
            
    # 3. User-User Similarity Index
    print("Computing User-User Similarity...")
    C = defaultdict(lambda: defaultdict(int))
    
    for mid, users in tqdm(item_user_index.items(), desc="Co-occurrences"):
        users_list = list(users)
        for i in range(len(users_list)):
            u = users_list[i]
            for j in range(i + 1, len(users_list)):
                v = users_list[j]
                C[u][v] += 1
                C[v][u] += 1
                
    user_sim_index = {}
    
    for u, related_users in tqdm(C.items(), desc="Sim Scores"):
        scores = []
        len_u = len(user_item_set[u])
        
        for v, count in related_users.items():
            len_v = len(user_item_set[v])
            sim = count / math.sqrt(len_u * len_v)
            scores.append((v, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        user_sim_index[u] = scores[:sim_top_k]
        
    return user_sim_index, user_item_history, user_item_set

def recommend(user_id, user_sim_index, user_item_history, user_item_set, top_k_sim=20, last_n=20, top_n_rec=100):
    """
    UserCF 推荐逻辑
    1. 找到 Top-K 相似用户
    2. 获取每个相似用户的 Last-N 物品
    3. 预估分数并推荐
    """
    if user_id not in user_sim_index:
        return []
        
    similar_users = user_sim_index[user_id][:top_k_sim]
    
    candidates = defaultdict(float)
    watched_items = user_item_set.get(user_id, set())
    
    for v, sim_score in similar_users:
        # 获取相似用户 v 的近期感兴趣物品 (Last-n)
        # user_item_history 已经是倒序的，直接取前 n 个
        v_history = user_item_history.get(v, [])[:last_n]
        
        for mid, rating in v_history:
            if mid in watched_items:
                continue
                
            # 兴趣分数公式: Sim(u, v) * Rating(v, mid)
            candidates[mid] += sim_score * rating
            
    # 返回 Top-N
    recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n_rec]
    return [mid for mid, score in recs]

def save_experiment_results(metrics, metadata, output_dir=OUTPUT_DIR):
    """保存评估实验数据到 output 目录，该目录已被 .gitignore 忽略。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": "usercf",
        "timestamp": timestamp,
        "metadata": metadata,
        "metrics": metrics,
    }
    json_path = os.path.join(output_dir, f"experiment_{timestamp}.json")
    csv_path = os.path.join(output_dir, "experiment_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    row = {
        "timestamp": timestamp,
        "model": "usercf",
        **metadata,
    }
    for k, values in metrics.items():
        row[f"HR@{k}"] = values["hr"]
        row[f"NDCG@{k}"] = values["ndcg"]

    pd.DataFrame([row]).to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
        encoding="utf-8",
    )
    print(f"Experiment data saved to {json_path}")


def calculate_metrics(test_df, user_sim_index, user_item_history, user_item_set, ks=[3, 5, 10]):
    """计算指标"""
    print("\nCalculating metrics...")
    
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    total_users = 0
    
    test_data = test_df[['UserID', 'MovieID']].values
    
    for uid, target_mid in tqdm(test_data, desc="Evaluating"):
        total_users += 1
        
        # 生成推荐列表
        max_k = max(ks)
        # 使用题目建议的参数逻辑: Top-K 相似用户, Last-N 物品
        rec_list = recommend(
            uid, 
            user_sim_index, 
            user_item_history, 
            user_item_set,
            top_k_sim=50,  # 相似用户数
            last_n=50,     # 每个相似用户的物品数
            top_n_rec=max_k
        )
        
        for k in ks:
            top_k_recs = rec_list[:k]
            if target_mid in top_k_recs:
                hits[k] += 1
                rank = top_k_recs.index(target_mid)
                ndcgs[k] += 1.0 / math.log2(rank + 2)
                
    print("-" * 40)
    print("UserCF Evaluation Results:")
    print("-" * 40)
    metrics = {}
    for k in ks:
        hr = hits[k] / total_users
        ndcg = ndcgs[k] / total_users
        metrics[k] = {"hr": hr, "ndcg": ndcg}
        print(f"HR@{k}: {hr:.4f}")
        print(f"NDCG@{k}: {ndcg:.4f}")
    print("-" * 40)
    return metrics

def main():
    # 1. 加载并划分数据
    train_df, test_df = load_and_split_data()
    
    # 2. 构建索引 (Sim Top-K = 100, 为了保证足够的候选)
    user_sim_index, user_item_history, user_item_set = build_indices(train_df, sim_top_k=100)
    
    # 3. 评估
    ks = [3, 5, 10]
    metrics = calculate_metrics(test_df, user_sim_index, user_item_history, user_item_set, ks=ks)
    save_experiment_results(
        metrics,
        metadata={
            **get_dataset_info(),
            "data_dir": DATA_DIR,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "sim_top_k": 100,
            "top_k_sim": 50,
            "last_n": 50,
            "ks": "|".join(map(str, ks)),
        },
    )

if __name__ == "__main__":
    main()




