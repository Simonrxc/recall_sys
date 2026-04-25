#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MovieLens 1M 数据集探索脚本
用于了解数据集的结构和内容
"""

import pandas as pd
import os
from datetime import datetime

# 数据集路径
DATA_DIR = "dataset/ml-1m"

def load_ratings():
    """加载评分数据"""
    print("=" * 60)
    print("加载评分数据 (ratings.dat)...")
    print("=" * 60)
    
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "ratings.dat"),
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )
    
    # 转换时间戳为日期时间
    ratings["DateTime"] = pd.to_datetime(ratings["Timestamp"], unit="s")
    
    print(f"总评分数量: {len(ratings):,}")
    print(f"用户数量: {ratings['UserID'].nunique():,}")
    print(f"电影数量: {ratings['MovieID'].nunique():,}")
    print(f"\n评分数据前5行:")
    print(ratings.head())
    print(f"\n评分统计:")
    print(ratings["Rating"].value_counts().sort_index())
    print(f"\n平均评分: {ratings['Rating'].mean():.2f}")
    print(f"评分时间范围: {ratings['DateTime'].min()} 到 {ratings['DateTime'].max()}")
    
    return ratings

def load_movies():
    """加载电影数据"""
    print("\n" + "=" * 60)
    print("加载电影数据 (movies.dat)...")
    print("=" * 60)
    
    movies = pd.read_csv(
        os.path.join(DATA_DIR, "movies.dat"),
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    
    print(f"总电影数量: {len(movies):,}")
    print(f"\n电影数据前10行:")
    print(movies.head(10))
    
    # 分析电影类型
    all_genres = []
    for genres in movies["Genres"]:
        if genres and genres != "(no genres listed)":
            all_genres.extend(genres.split("|"))
    
    genre_counts = pd.Series(all_genres).value_counts()
    print(f"\n电影类型统计 (前10):")
    print(genre_counts.head(10))
    
    return movies

def load_users():
    """加载用户数据"""
    print("\n" + "=" * 60)
    print("加载用户数据 (users.dat)...")
    print("=" * 60)
    
    users = pd.read_csv(
        os.path.join(DATA_DIR, "users.dat"),
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="latin-1"
    )
    
    print(f"总用户数量: {len(users):,}")
    print(f"\n用户数据前10行:")
    print(users.head(10))
    
    # 年龄分布
    age_labels = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    users["AgeGroup"] = users["Age"].map(age_labels)
    print(f"\n年龄分布:")
    print(users["AgeGroup"].value_counts().sort_index())
    
    print(f"\n性别分布:")
    print(users["Gender"].value_counts())
    
    return users

def analyze_data(ratings, movies, users):
    """分析数据集的关联信息"""
    print("\n" + "=" * 60)
    print("数据分析")
    print("=" * 60)
    
    # 合并数据
    ratings_with_movies = ratings.merge(movies, on="MovieID", how="left")
    ratings_with_users = ratings.merge(users, on="UserID", how="left")
    
    # 最受欢迎的电影（评分次数最多）
    print("\n最受欢迎的电影 (评分次数最多的前10部):")
    popular_movies = ratings.groupby("MovieID").size().sort_values(ascending=False).head(10)
    popular_movies_df = pd.DataFrame({
        "MovieID": popular_movies.index,
        "RatingCount": popular_movies.values
    }).merge(movies, on="MovieID")
    print(popular_movies_df[["Title", "Genres", "RatingCount"]])
    
    # 评分最高的电影（平均评分）
    print("\n评分最高的电影 (平均评分最高的前10部，至少100个评分):")
    movie_ratings = ratings.groupby("MovieID").agg({
        "Rating": ["mean", "count"]
    }).reset_index()
    movie_ratings.columns = ["MovieID", "AvgRating", "RatingCount"]
    top_rated = movie_ratings[movie_ratings["RatingCount"] >= 100].sort_values(
        "AvgRating", ascending=False
    ).head(10)
    top_rated_df = top_rated.merge(movies, on="MovieID")
    print(top_rated_df[["Title", "Genres", "AvgRating", "RatingCount"]])
    
    # 最活跃的用户
    print("\n最活跃的用户 (评分次数最多的前10个用户):")
    active_users = ratings.groupby("UserID").size().sort_values(ascending=False).head(10)
    active_users_df = pd.DataFrame({
        "UserID": active_users.index,
        "RatingCount": active_users.values
    }).merge(users, on="UserID")
    print(active_users_df[["UserID", "Gender", "AgeGroup", "RatingCount"]])
    
    # 不同性别的评分偏好
    print("\n不同性别的平均评分:")
    gender_ratings = ratings_with_users.groupby("Gender")["Rating"].agg(["mean", "count"])
    print(gender_ratings)
    
    # 不同年龄组的评分偏好
    print("\n不同年龄组的平均评分:")
    age_ratings = ratings_with_users.groupby("AgeGroup")["Rating"].agg(["mean", "count"])
    print(age_ratings.sort_index())

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MovieLens 1M 数据集探索")
    print("=" * 60)
    
    # 检查数据目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录 {DATA_DIR} 不存在！")
        return
    
    # 加载数据
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    
    # 分析数据
    analyze_data(ratings, movies, users)
    
    print("\n" + "=" * 60)
    print("探索完成！")
    print("=" * 60)
    print("\n提示: 你可以使用以下方式访问数据:")
    print("  - ratings: 包含所有评分信息")
    print("  - movies: 包含所有电影信息")
    print("  - users: 包含所有用户信息")
    print("\n示例: 查看用户1的所有评分")
    print("  user1_ratings = ratings[ratings['UserID'] == 1]")
    print("  user1_with_movies = user1_ratings.merge(movies, on='MovieID')")
    print("  print(user1_with_movies[['Title', 'Rating', 'DateTime']])")

if __name__ == "__main__":
    main()

