#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用户向量化模块
将 MovieLens 数据集中的用户转换为向量表示，用于召回系统
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 数据集路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIRS = [
    os.path.join(REPO_ROOT, "convert_dataset"),
    os.path.join(REPO_ROOT, "converted_dataset"),
]
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def resolve_data_dir():
    """定位 convert_dataset.py 生成的统一数据目录。"""
    env_data_dir = os.environ.get("USER2EMB_DATA_DIR") or os.environ.get("RECALL_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(env_data_dir)

    for data_dir in DEFAULT_DATA_DIRS:
        if os.path.isdir(data_dir):
            return data_dir

    return DEFAULT_DATA_DIRS[0]


DATA_DIR = resolve_data_dir()


def load_data():
    """加载 convert_dataset.py 输出的统一 CSV 数据。"""
    print(f"正在从 {DATA_DIR} 加载统一格式数据...")
    users_path = os.path.join(DATA_DIR, "users.csv")
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    missing_files = [
        path for path in [users_path, movies_path, ratings_path]
        if not os.path.exists(path)
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"未找到转换后的数据文件: {missing}\n"
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
    users = pd.read_csv(users_path).rename(
        columns={
            "user_id": "UserID",
            "gender": "Gender",
            "age": "Age",
            "occupation": "Occupation",
            "zip_code": "Zip-code",
        }
    )
    movies = pd.read_csv(movies_path).rename(
        columns={
            "movie_id": "MovieID",
            "title": "Title",
            "genres": "Genres",
        }
    )

    ratings = ratings[["UserID", "MovieID", "Rating", "Timestamp"]].copy()
    users = users[["UserID", "Gender", "Age", "Occupation", "Zip-code"]].copy()
    movies = movies[["MovieID", "Title", "Genres"]].copy()

    users["Gender"] = users["Gender"].fillna("Unknown").astype(str)
    users["Age"] = users["Age"].fillna(0)
    users["Occupation"] = users["Occupation"].fillna("Unknown").astype(str)
    users["Zip-code"] = users["Zip-code"].fillna("").astype(str)
    movies["Title"] = movies["Title"].fillna("").astype(str)
    movies["Genres"] = movies["Genres"].fillna("(no genres listed)").astype(str)
    
    print(f"加载完成: {len(ratings):,} 条评分, {len(users):,} 个用户, {len(movies):,} 部电影")
    return ratings, users, movies


def extract_user_features(ratings, users, movies):
    """提取用户特征"""
    print("\n正在提取用户特征...")
    
    # 合并数据
    ratings_with_movies = ratings.merge(movies, on="MovieID", how="left")
    
    # 用户行为统计特征
    user_stats = ratings.groupby("UserID").agg({
        "Rating": ["mean", "std", "count", "min", "max"],
        "Timestamp": ["min", "max"]
    }).reset_index()
    
    user_stats.columns = ["UserID", "AvgRating", "RatingStd", "RatingCount", 
                          "MinRating", "MaxRating", "FirstRatingTime", "LastRatingTime"]
    
    # 计算用户活跃天数
    user_stats["ActiveDays"] = (
        user_stats["LastRatingTime"] - user_stats["FirstRatingTime"]
    ) / (24 * 3600)  # 转换为天数
    user_stats["ActiveDays"] = user_stats["ActiveDays"].fillna(0)
    
    # 用户评分分布特征（各评分的比例）
    rating_dist = ratings.groupby(["UserID", "Rating"]).size().unstack(fill_value=0)
    rating_dist = rating_dist.div(rating_dist.sum(axis=1), axis=0)  # 归一化
    rating_dist.columns = [f"Rating_{col}_Ratio" for col in rating_dist.columns]
    rating_dist = rating_dist.reset_index()
    
    # 用户最喜爱的电影类型
    user_genres = []
    for user_id in users["UserID"]:
        user_ratings = ratings_with_movies[ratings_with_movies["UserID"] == user_id]
        if len(user_ratings) > 0:
            # 获取用户评分过的所有类型
            all_genres = []
            for genres in user_ratings["Genres"]:
                if pd.notna(genres) and genres != "(no genres listed)":
                    all_genres.extend(genres.split("|"))
            
            # 计算各类型的加权平均评分（按评分加权）
            genre_scores = {}
            for idx, row in user_ratings.iterrows():
                if pd.notna(row["Genres"]) and row["Genres"] != "(no genres listed)":
                    for genre in row["Genres"].split("|"):
                        if genre not in genre_scores:
                            genre_scores[genre] = []
                        genre_scores[genre].append(row["Rating"])
            
            # 计算每个类型的平均评分
            genre_avg_ratings = {
                genre: np.mean(scores) 
                for genre, scores in genre_scores.items()
            }
            
            # 获取前5个最喜爱的类型
            top_genres = sorted(genre_avg_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
            user_genres.append({
                "UserID": user_id,
                "TopGenre1": top_genres[0][0] if len(top_genres) > 0 else "Unknown",
                "TopGenre2": top_genres[1][0] if len(top_genres) > 1 else "Unknown",
                "TopGenre3": top_genres[2][0] if len(top_genres) > 2 else "Unknown",
            })
        else:
            user_genres.append({
                "UserID": user_id,
                "TopGenre1": "Unknown",
                "TopGenre2": "Unknown",
                "TopGenre3": "Unknown",
            })
    
    user_genres_df = pd.DataFrame(user_genres)
    
    # 合并所有特征
    user_features = users.merge(user_stats, on="UserID", how="left")
    user_features = user_features.merge(rating_dist, on="UserID", how="left")
    user_features = user_features.merge(user_genres_df, on="UserID", how="left")
    
    # 填充缺失值
    user_features = user_features.fillna(0)
    
    print(f"特征提取完成，共 {len(user_features.columns)} 个特征")
    return user_features


def encode_categorical_features(user_features, ratings, movies):
    """编码分类特征"""
    print("\n正在编码分类特征...")
    
    # 性别编码
    gender_encoder = LabelEncoder()
    user_features["Gender_Encoded"] = gender_encoder.fit_transform(user_features["Gender"])
    
    # 年龄编码（已经是数值，但可以保持原样或进行分桶）
    # 年龄已经是数值，直接使用
    
    # 职业编码
    occupation_encoder = LabelEncoder()
    user_features["Occupation_Encoded"] = occupation_encoder.fit_transform(
        user_features["Occupation"].astype(str)
    )
    
    # 获取所有电影类型
    all_genres = set()
    for genres in movies["Genres"]:
        if pd.notna(genres) and genres != "(no genres listed)":
            all_genres.update(genres.split("|"))
    
    # 为每个用户计算所有类型的评分统计
    ratings_with_movies = ratings.merge(movies, on="MovieID", how="left")
    
    genre_stats = []
    for user_id in user_features["UserID"]:
        user_ratings = ratings_with_movies[ratings_with_movies["UserID"] == user_id]
        
        genre_dict = {}
        for genre in all_genres:
            genre_dict[f"Genre_{genre}_Count"] = 0
            genre_dict[f"Genre_{genre}_AvgRating"] = 0.0
        
        # 统计每个类型的评分
        genre_ratings = {}
        for idx, row in user_ratings.iterrows():
            if pd.notna(row["Genres"]) and row["Genres"] != "(no genres listed)":
                for genre in row["Genres"].split("|"):
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    genre_ratings[genre].append(row["Rating"])
        
        # 计算每个类型的统计信息
        for genre in all_genres:
            if genre in genre_ratings:
                genre_dict[f"Genre_{genre}_Count"] = len(genre_ratings[genre])
                genre_dict[f"Genre_{genre}_AvgRating"] = np.mean(genre_ratings[genre])
        
        genre_dict["UserID"] = user_id
        genre_stats.append(genre_dict)
    
    genre_stats_df = pd.DataFrame(genre_stats)
    user_features = user_features.merge(genre_stats_df, on="UserID", how="left")
    
    # 填充缺失值
    genre_cols = [col for col in user_features.columns if col.startswith("Genre_")]
    user_features[genre_cols] = user_features[genre_cols].fillna(0)
    
    # 保存编码器
    encoders = {
        "gender": gender_encoder,
        "occupation": occupation_encoder,
    }
    
    return user_features, encoders


def create_user_embeddings(user_features, embedding_dim=128):
    """创建用户向量"""
    print(f"\n正在创建用户向量 (维度: {embedding_dim})...")
    
    # 选择数值特征
    numeric_features = [
        "Gender_Encoded",
        "Age",
        "Occupation_Encoded",
        "AvgRating",
        "RatingStd",
        "RatingCount",
        "MinRating",
        "MaxRating",
        "ActiveDays",
    ]
    
    # 添加评分分布特征
    rating_ratio_features = [col for col in user_features.columns 
                            if col.startswith("Rating_") and col.endswith("_Ratio")]
    numeric_features.extend(rating_ratio_features)
    
    # 添加类型特征（包括数量和平均评分）
    genre_features = [col for col in user_features.columns 
                     if col.startswith("Genre_")]
    numeric_features.extend(genre_features)
    
    # 提取特征矩阵
    feature_matrix = user_features[numeric_features].values
    
    print(f"原始特征维度: {feature_matrix.shape[1]}")
    
    # 标准化特征
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # 使用 PCA 降维到目标维度
    if feature_matrix_scaled.shape[1] > embedding_dim:
        pca = PCA(n_components=embedding_dim)
        user_embeddings = pca.fit_transform(feature_matrix_scaled)
        print(f"使用 PCA 降维: {feature_matrix_scaled.shape[1]} -> {embedding_dim}")
        print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    elif feature_matrix_scaled.shape[1] == embedding_dim:
        # 如果特征维度等于目标维度，直接使用
        user_embeddings = feature_matrix_scaled
        print(f"特征维度等于目标维度，直接使用: {embedding_dim}")
    else:
        # 如果特征维度小于目标维度，先使用PCA保留所有信息，然后使用随机投影扩展
        # 或者直接使用所有特征，然后用零填充
        # 这里我们使用零填充，但也可以考虑使用随机投影
        from sklearn.random_projection import GaussianRandomProjection
        
        # 先保留所有原始特征
        base_dim = feature_matrix_scaled.shape[1]
        remaining_dim = embedding_dim - base_dim
        
        # 使用随机投影将特征扩展到目标维度
        if remaining_dim > 0:
            rp = GaussianRandomProjection(n_components=remaining_dim, random_state=42)
            extended_features = rp.fit_transform(feature_matrix_scaled)
            user_embeddings = np.hstack([feature_matrix_scaled, extended_features])
            print(f"特征扩展: {base_dim} -> {embedding_dim} (使用随机投影)")
        else:
            user_embeddings = feature_matrix_scaled
    
    # 创建用户ID到向量的映射
    user_id_to_embedding = dict(zip(user_features["UserID"], user_embeddings))
    
    print(f"用户向量创建完成: {len(user_id_to_embedding):,} 个用户")
    print(f"向量维度: {user_embeddings.shape[1]}")
    print(f"向量示例 (用户ID=1): {user_embeddings[0][:10]}...")
    
    return user_embeddings, user_id_to_embedding, scaler


def save_embeddings(user_embeddings, user_id_to_embedding, user_features, scaler, encoders):
    """保存用户向量和相关信息"""
    print("\n正在保存结果...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存用户向量矩阵
    np.save(os.path.join(OUTPUT_DIR, "user_embeddings.npy"), user_embeddings)
    print(f"✓ 保存用户向量矩阵: {OUTPUT_DIR}/user_embeddings.npy")
    
    # 保存用户ID到向量的映射
    with open(os.path.join(OUTPUT_DIR, "user_id_to_embedding.pkl"), "wb") as f:
        pickle.dump(user_id_to_embedding, f)
    print(f"✓ 保存用户ID映射: {OUTPUT_DIR}/user_id_to_embedding.pkl")
    
    # 保存用户特征（用于后续分析）
    user_features.to_csv(
        os.path.join(OUTPUT_DIR, "user_features.csv"),
        index=False,
        encoding="utf-8"
    )
    print(f"✓ 保存用户特征: {OUTPUT_DIR}/user_features.csv")
    
    # 保存预处理对象
    with open(os.path.join(OUTPUT_DIR, "preprocessing_objects.pkl"), "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "encoders": encoders
        }, f)
    print(f"✓ 保存预处理对象: {OUTPUT_DIR}/preprocessing_objects.pkl")
    
    # 保存用户ID列表（保持顺序）
    user_ids = user_features["UserID"].values
    np.save(os.path.join(OUTPUT_DIR, "user_ids.npy"), user_ids)
    print(f"✓ 保存用户ID列表: {OUTPUT_DIR}/user_ids.npy")


def load_user_embedding(user_id):
    """加载指定用户的向量（示例函数）"""
    embedding_path = os.path.join(OUTPUT_DIR, "user_id_to_embedding.pkl")
    if os.path.exists(embedding_path):
        with open(embedding_path, "rb") as f:
            user_id_to_embedding = pickle.load(f)
        return user_id_to_embedding.get(user_id, None)
    return None


def main():
    """主函数"""
    print("=" * 60)
    print("用户向量化 - MovieLens 数据集")
    print("=" * 60)
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录 {DATA_DIR} 不存在！")
        return
    
    # 加载数据
    ratings, users, movies = load_data()
    
    # 提取用户特征
    user_features = extract_user_features(ratings, users, movies)
    
    # 编码分类特征
    user_features, encoders = encode_categorical_features(user_features, ratings, movies)
    
    # 创建用户向量
    user_embeddings, user_id_to_embedding, scaler = create_user_embeddings(
        user_features, 
        embedding_dim=128
    )
    
    # 保存结果
    save_embeddings(user_embeddings, user_id_to_embedding, user_features, scaler, encoders)
    
    print("\n" + "=" * 60)
    print("用户向量化完成！")
    print("=" * 60)
    print("\n输出文件:")
    print(f"  - {OUTPUT_DIR}/user_embeddings.npy: 用户向量矩阵")
    print(f"  - {OUTPUT_DIR}/user_id_to_embedding.pkl: 用户ID到向量的映射")
    print(f"  - {OUTPUT_DIR}/user_features.csv: 用户特征表")
    print(f"  - {OUTPUT_DIR}/preprocessing_objects.pkl: 预处理对象")
    print(f"  - {OUTPUT_DIR}/user_ids.npy: 用户ID列表")
    print("\n使用示例:")
    print("  from user_embedding import load_user_embedding")
    print("  embedding = load_user_embedding(1)  # 获取用户1的向量")


if __name__ == "__main__":
    main()

