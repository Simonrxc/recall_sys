#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import pickle
import random

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
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载 convert_dataset.py 输出的统一评分 CSV。"""
    print(f"Loading converted ratings from {DATA_DIR}...")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"未找到转换后的评分文件: {ratings_path}\n"
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
    return ratings

def split_users(ratings, test_ratio=0.2, seed=42):
    """
    随机分出一部分用户作为测试集
    这里的意思可能是：我们只对这部分用户进行评估，或者说把这些用户的部分行为作为测试集。
    通常 UserCF 是基于全量用户历史构建相似度，然后在测试集上预测。
    
    这里我们采用：
    1. 随机选 test_ratio 的用户作为 "测试用户"。
    2. 对于所有用户，使用他们的全量历史构建 User-Item 索引和 User-User 索引。
       (在真实的离线评估中，应该把测试用户的最后行为 mask 掉，这里为了简化演示，先建立全量索引)
    """
    all_users = ratings['UserID'].unique()
    random.seed(seed)
    # 随机选择测试用户
    test_users = set(random.sample(list(all_users), int(len(all_users) * test_ratio)))
    
    print(f"Total users: {len(all_users)}, Test users: {len(test_users)}")
    return test_users

def build_user_item_index(ratings):
    """
    建立【用户到物品】的索引
    Returns:
        user_item_index: {user_id: {movie_id}} (set for fast intersection)
        user_item_rating: {user_id: [(movie_id, rating), ...]}
    """
    print("Building User-Item Index...")
    user_item_index = defaultdict(set)
    user_item_rating = defaultdict(list)
    
    # 转换为 numpy 加速
    data = ratings[['UserID', 'MovieID', 'Rating']].values
    
    for row in tqdm(data, desc="Indexing"):
        uid, mid, rating = int(row[0]), int(row[1]), float(row[2])
        # 认为 Rating >= 0 的都是感兴趣，或者可以设置阈值
        user_item_index[uid].add(mid)
        user_item_rating[uid].append((mid, rating))
        
    return user_item_index, user_item_rating

def build_item_user_index(user_item_index):
    """
    建立倒排索引【物品到用户】，用于加速相似度计算
    Returns:
        item_user_index: {movie_id: {user_id}}
    """
    print("Building Item-User Index (Inverted Index)...")
    item_user_index = defaultdict(set)
    for uid, items in user_item_index.items():
        for mid in items:
            item_user_index[mid].add(uid)
    return item_user_index

def build_user_user_similarity(user_item_index, item_user_index, top_k=20):
    """
    建立【用户到用户】的相似度索引
    Sim(u, v) = |I_u n I_v| / sqrt(|I_u| * |I_v|)
    """
    print("Computing User-User Similarity...")
    
    # 1. 计算共现次数 C[u][v]
    # 使用倒排索引加速：只计算有过共同物品的用户对
    C = defaultdict(lambda: defaultdict(int))
    
    for mid, users in tqdm(item_user_index.items(), desc="Counting co-occurrences"):
        users_list = list(users)
        # 对于该物品下的所有用户两两组合
        for i in range(len(users_list)):
            u = users_list[i]
            for j in range(i + 1, len(users_list)):
                v = users_list[j]
                C[u][v] += 1
                C[v][u] += 1
                
    # 2. 计算相似度并保留 Top-K
    print("Calculating similarity scores...")
    user_sim_index = {}
    
    for u, related_users in tqdm(C.items(), desc="Calculating scores"):
        scores = []
        len_u = len(user_item_index[u])
        
        for v, count in related_users.items():
            len_v = len(user_item_index[v])
            # 公式: count / sqrt(len_u * len_v)
            sim = count / math.sqrt(len_u * len_v)
            scores.append((v, sim))
            
        # 排序并取 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        user_sim_index[u] = scores[:top_k]
        
    return user_sim_index

def save_results(user_sim_index, user_item_rating):
    """保存结果"""
    print(f"Saving results to {OUTPUT_DIR}...")
    with open(os.path.join(OUTPUT_DIR, "user_sim_index.pkl"), "wb") as f:
        pickle.dump(user_sim_index, f)
        
    with open(os.path.join(OUTPUT_DIR, "user_item_rating.pkl"), "wb") as f:
        pickle.dump(user_item_rating, f)
    print("Done.")

def main():
    # 1. 加载数据
    ratings = load_data()
    
    # 2. 随机分出一部分测试用户 (这里主要是标记，实际构建索引使用全量数据)
    test_users = split_users(ratings)
    
    # 3. 建立【用户到物品】索引
    user_item_index, user_item_rating = build_user_item_index(ratings)
    
    # 4. 建立倒排索引 (加速用)
    item_user_index = build_item_user_index(user_item_index)
    
    # 5. 建立【用户到用户】索引
    user_sim_index = build_user_user_similarity(user_item_index, item_user_index, top_k=20)
    
    # 6. 保存
    save_results(user_sim_index, user_item_rating)
    
    # 7. 简单展示一个测试用户的相似用户
    if test_users:
        sample_user = list(test_users)[0]
        print(f"\nUser {sample_user} Top-10 Similar Users:")
        if sample_user in user_sim_index:
            for sim_user, score in user_sim_index[sample_user][:10]:
                print(f"  - User {sim_user}: Similarity = {score:.4f}")
        else:
            print("  No similar users found (maybe inactive user).")

if __name__ == "__main__":
    main()




