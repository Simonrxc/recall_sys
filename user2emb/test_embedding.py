#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试用户向量化结果
"""

import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "./output"


def load_embeddings():
    """加载用户向量"""
    # 加载用户ID到向量的映射
    with open(os.path.join(OUTPUT_DIR, "user_id_to_embedding.pkl"), "rb") as f:
        user_id_to_embedding = pickle.load(f)
    
    # 加载用户ID列表
    user_ids = np.load(os.path.join(OUTPUT_DIR, "user_ids.npy"))
    
    # 加载向量矩阵
    embeddings = np.load(os.path.join(OUTPUT_DIR, "user_embeddings.npy"))
    
    return user_id_to_embedding, user_ids, embeddings


def find_similar_users(user_id, user_id_to_embedding, top_k=10):
    """找到与指定用户最相似的用户"""
    if user_id not in user_id_to_embedding:
        print(f"用户 {user_id} 不存在")
        return None
    
    target_embedding = user_id_to_embedding[user_id].reshape(1, -1)
    
    # 计算与所有用户的相似度
    similarities = []
    for uid, emb in user_id_to_embedding.items():
        if uid != user_id:
            sim = cosine_similarity(target_embedding, emb.reshape(1, -1))[0][0]
            similarities.append((uid, sim))
    
    # 排序并返回top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def main():
    """主函数"""
    print("=" * 60)
    print("测试用户向量化结果")
    print("=" * 60)
    
    # 加载向量
    user_id_to_embedding, user_ids, embeddings = load_embeddings()
    
    print(f"\n加载完成:")
    print(f"  - 用户数量: {len(user_id_to_embedding):,}")
    print(f"  - 向量维度: {embeddings.shape[1]}")
    print(f"  - 向量矩阵形状: {embeddings.shape}")
    
    # 测试几个用户
    test_users = [1, 2, 3, 100, 1000]
    
    print("\n" + "=" * 60)
    print("用户向量示例")
    print("=" * 60)
    for user_id in test_users:
        if user_id in user_id_to_embedding:
            emb = user_id_to_embedding[user_id]
            print(f"\n用户 {user_id}:")
            print(f"  向量维度: {emb.shape}")
            print(f"  向量前10维: {emb[:10]}")
            print(f"  向量统计: min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}, std={emb.std():.4f}")
    
    # 找到相似用户
    print("\n" + "=" * 60)
    print("相似用户查找示例")
    print("=" * 60)
    for user_id in [1, 100]:
        if user_id in user_id_to_embedding:
            print(f"\n与用户 {user_id} 最相似的10个用户:")
            similar_users = find_similar_users(user_id, user_id_to_embedding, top_k=10)
            if similar_users:
                for i, (uid, sim) in enumerate(similar_users, 1):
                    print(f"  {i}. 用户 {uid}: 相似度 = {sim:.4f}")
    
    # 向量统计
    print("\n" + "=" * 60)
    print("向量统计信息")
    print("=" * 60)
    all_embeddings = np.array(list(user_id_to_embedding.values()))
    print(f"所有用户向量统计:")
    print(f"  最小值: {all_embeddings.min():.4f}")
    print(f"  最大值: {all_embeddings.max():.4f}")
    print(f"  平均值: {all_embeddings.mean():.4f}")
    print(f"  标准差: {all_embeddings.std():.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()








