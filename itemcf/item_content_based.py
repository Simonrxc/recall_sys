#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# 配置路径
DATA_DIR = "../dataset/ml-1m"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载数据"""
    print("Loading data...")
    # 加载电影数据
    movies = pd.read_csv(
        os.path.join(DATA_DIR, "movies.dat"),
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    
    # 加载评分数据
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "ratings.dat"),
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )
    return movies, ratings

def train_genre_embeddings(movies, vector_size=32, window=5, min_count=1):
    """
    步骤1: 对 Genres 进行 Word2vec 训练
    """
    print("Training Genre Word2vec...")
    
    # 准备语料：将每部电影的 Genres 拆分为列表，作为"句子"
    # 例如: "Animation|Children's|Comedy" -> ['Animation', "Children's", 'Comedy']
    sentences = [genres.split('|') for genres in movies['Genres']]
    
    # 训练模型
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, seed=42)
    
    # 获取每个 Genre 的向量
    genre_vectors = {genre: model.wv[genre] for genre in model.wv.index_to_key}
    
    print(f"Trained embeddings for {len(genre_vectors)} genres.")
    return genre_vectors

def compute_movie_embeddings(movies, genre_vectors, vector_size=32):
    """
    步骤2: 计算电影 Embedding (Genres Embedding 求平均)
    """
    print("Computing Movie Embeddings...")
    
    movie_embeddings = {}
    
    for _, row in movies.iterrows():
        movie_id = row['MovieID']
        genres = row['Genres'].split('|')
        
        # 获取该电影所有 Genre 的向量
        vectors = [genre_vectors[g] for g in genres if g in genre_vectors]
        
        if vectors:
            # 求平均
            avg_vector = np.mean(vectors, axis=0)
            # 归一化 (方便后续计算 Cosine Similarity)
            norm = np.linalg.norm(avg_vector)
            if norm > 0:
                avg_vector = avg_vector / norm
            movie_embeddings[movie_id] = avg_vector
        else:
            # 如果没有对应的 Genre 向量（极少情况），用零向量填充
            movie_embeddings[movie_id] = np.zeros(vector_size)
            
    return movie_embeddings

def build_item_item_index(movie_embeddings):
    """
    步骤3: 建立【物品到物品】索引 (计算相似度)
    这里为了简单直观，计算全量相似度矩阵。对于海量数据应使用 FAISS。
    """
    print("Building Item-Item Similarity Index...")
    
    movie_ids = list(movie_embeddings.keys())
    # 转换为矩阵 (N_movies, Vector_Size)
    embedding_matrix = np.array([movie_embeddings[mid] for mid in movie_ids])
    
    # 计算余弦相似度矩阵 (N, N)
    # 因为我们在 compute_movie_embeddings 里已经做了归一化，所以 dot product 就是 cosine similarity
    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    
    # 构建快速查找索引
    # item_sim_index[movie_id] = [(similar_movie_id, score), ...]
    item_sim_index = {}
    
    # 保存 top_k 相似物品
    top_k = 20
    
    for i, mid in enumerate(movie_ids):
        # 获取第 i 个电影的所有相似度
        scores = similarity_matrix[i]
        
        # 获取 top_k 索引 (排除自己)
        # argsort 返回从小到大的索引，取最后 k+1 个，然后逆序
        top_indices = np.argsort(scores)[-(top_k+1):][::-1]
        
        similar_items = []
        for idx in top_indices:
            sim_mid = movie_ids[idx]
            if sim_mid != mid:
                similar_items.append((sim_mid, float(scores[idx])))
                if len(similar_items) >= top_k:
                    break
        
        item_sim_index[mid] = similar_items
        
    return item_sim_index

def build_user_history_index(ratings):
    """
    步骤4: 建立【用户到物品】索引 (用户近期感兴趣的电影)
    """
    print("Building User History Index...")
    
    # 按时间排序
    ratings_sorted = ratings.sort_values(by=['UserID', 'Timestamp'], ascending=[True, False])
    
    user_history_index = defaultdict(list)
    
    # 聚合每个用户的历史
    # 格式: {user_id: [(movie_id, rating, timestamp), ...]}
    # GroupBy 比较慢，这里使用迭代优化
    current_user = None
    temp_history = []
    
    # 转换为 numpy 数组加速迭代
    # columns: UserID, MovieID, Rating, Timestamp
    data_values = ratings_sorted[['UserID', 'MovieID', 'Rating', 'Timestamp']].values
    
    for row in data_values:
        uid, mid, rating, ts = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        
        user_history_index[uid].append((mid, rating, ts))
            
    return user_history_index

def save_results(movie_embeddings, item_sim_index, user_history_index, genre_vectors):
    """保存所有结果"""
    print(f"Saving results to {OUTPUT_DIR}...")
    
    with open(os.path.join(OUTPUT_DIR, "movie_embeddings.pkl"), "wb") as f:
        pickle.dump(movie_embeddings, f)
        
    with open(os.path.join(OUTPUT_DIR, "item_sim_index.pkl"), "wb") as f:
        pickle.dump(item_sim_index, f)
        
    with open(os.path.join(OUTPUT_DIR, "user_history_index.pkl"), "wb") as f:
        pickle.dump(user_history_index, f)
        
    with open(os.path.join(OUTPUT_DIR, "genre_vectors.pkl"), "wb") as f:
        pickle.dump(genre_vectors, f)
        
    print("Done.")

def recommend_for_user(user_id, user_history_index, item_sim_index, movies_df, top_n=10):
    """
    示例: 基于内容的 ItemCF 推荐
    """
    if user_id not in user_history_index:
        print(f"User {user_id} not found in history.")
        return []
    
    # 获取用户最近看过的 N 个高分电影 (比如 Rating >= 4)
    history = user_history_index[user_id]
    # 取最近 5 部喜欢的电影
    recent_liked = [item for item in history if item[1] >= 4.0][:5]
    
    print(f"\nUser {user_id} recent liked movies:")
    movie_titles = movies_df.set_index('MovieID')['Title'].to_dict()
    for mid, r, _ in recent_liked:
        print(f"  - {movie_titles.get(mid, mid)} (Rating: {r})")
        
    # 简单的推荐逻辑: 
    # 对于每个喜欢的电影，找最相似的电影，累加分数
    candidates = defaultdict(float)
    
    watched_mids = set(item[0] for item in history)
    
    for mid, rating, _ in recent_liked:
        # 获取该电影的相似电影
        if mid in item_sim_index:
            sim_items = item_sim_index[mid]
            for sim_mid, score in sim_items:
                if sim_mid not in watched_mids:
                    # 分数 = 相似度 * 用户对原物品的评分
                    candidates[sim_mid] += score * rating
    
    # 排序
    recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\nRecommendations for User {user_id}:")
    for mid, score in recs:
        print(f"  - {movie_titles.get(mid, mid)} (Score: {score:.4f})")
        
    return recs

if __name__ == "__main__":
    # 1. 加载数据
    movies, ratings = load_data()
    
    # 2. 训练 Genre Word2vec
    genre_vectors = train_genre_embeddings(movies, vector_size=32)
    
    # 3. 计算 Movie Embedding
    movie_embeddings = compute_movie_embeddings(movies, genre_vectors, vector_size=32)
    
    # 4. 建立 Item-Item 索引
    item_sim_index = build_item_item_index(movie_embeddings)
    
    # 5. 建立 User-Item 历史索引
    user_history_index = build_user_history_index(ratings)
    
    # 6. 保存结果
    save_results(movie_embeddings, item_sim_index, user_history_index, genre_vectors)
    
    # 7. 测试推荐
    # 随机选一个用户进行测试
    test_user_id = 1
    recommend_for_user(test_user_id, user_history_index, item_sim_index, movies)

