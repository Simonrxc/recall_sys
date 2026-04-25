# 用户向量化模块

本模块用于将 MovieLens 数据集中的用户转换为向量表示，用于召回系统。

## 功能特点

1. **用户特征提取**
   - 基础特征：性别、年龄、职业
   - 行为特征：平均评分、评分标准差、评分数量、活跃天数等
   - 评分分布：各评分等级的比例
   - 类型偏好：用户对各电影类型的评分统计

2. **特征编码**
   - 分类特征编码（性别、职业）
   - 电影类型特征提取（所有类型的评分统计）

3. **向量生成**
   - 特征标准化
   - 使用 PCA 或随机投影生成固定维度的用户向量（默认128维）

## 文件说明

- `user_embedding.py`: 主要的用户向量化脚本
- `test_embedding.py`: 测试脚本，用于验证向量化结果
- `output/`: 输出目录
  - `user_embeddings.npy`: 用户向量矩阵 (N×128)
  - `user_id_to_embedding.pkl`: 用户ID到向量的字典映射
  - `user_features.csv`: 用户特征表
  - `preprocessing_objects.pkl`: 预处理对象（标准化器、编码器等）
  - `user_ids.npy`: 用户ID列表（保持顺序）

## 使用方法

### 1. 生成用户向量

```bash
cd user2emb
python user_embedding.py
```

### 2. 在代码中使用

```python
import pickle
import numpy as np

# 加载用户向量
with open("output/user_id_to_embedding.pkl", "rb") as f:
    user_id_to_embedding = pickle.load(f)

# 获取用户1的向量
user1_embedding = user_id_to_embedding[1]
print(f"用户1的向量维度: {user1_embedding.shape}")
print(f"用户1的向量: {user1_embedding[:10]}...")

# 或者加载整个向量矩阵
embeddings = np.load("output/user_embeddings.npy")
user_ids = np.load("output/user_ids.npy")
print(f"向量矩阵形状: {embeddings.shape}")  # (6040, 128)
```

### 3. 计算用户相似度

```python
from sklearn.metrics.pairwise import cosine_similarity

# 获取两个用户的向量
user1_emb = user_id_to_embedding[1].reshape(1, -1)
user2_emb = user_id_to_embedding[2].reshape(1, -1)

# 计算余弦相似度
similarity = cosine_similarity(user1_emb, user2_emb)[0][0]
print(f"用户1和用户2的相似度: {similarity:.4f}")
```

### 4. 查找相似用户

```python
def find_similar_users(user_id, user_id_to_embedding, top_k=10):
    """找到与指定用户最相似的用户"""
    target_embedding = user_id_to_embedding[user_id].reshape(1, -1)
    
    similarities = []
    for uid, emb in user_id_to_embedding.items():
        if uid != user_id:
            sim = cosine_similarity(target_embedding, emb.reshape(1, -1))[0][0]
            similarities.append((uid, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 查找与用户1最相似的10个用户
similar_users = find_similar_users(1, user_id_to_embedding, top_k=10)
for uid, sim in similar_users:
    print(f"用户 {uid}: 相似度 = {sim:.4f}")
```

## 特征说明

用户向量包含以下类型的特征：

1. **基础特征** (3维)
   - 性别编码
   - 年龄
   - 职业编码

2. **行为统计特征** (6维)
   - 平均评分
   - 评分标准差
   - 评分数量
   - 最小评分
   - 最大评分
   - 活跃天数

3. **评分分布特征** (5维)
   - 各评分等级（1-5星）的比例

4. **类型偏好特征** (36维)
   - 18种电影类型 × 2（数量 + 平均评分）
   - 包括：Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

总计约50个原始特征，通过标准化和随机投影扩展到128维向量。

## 测试

运行测试脚本验证向量化结果：

```bash
python test_embedding.py
```

## 依赖库

- pandas
- numpy
- scikit-learn

## 注意事项

1. 确保数据集路径正确（默认：`../dataset/ml-1m`）
2. 输出目录会自动创建
3. 向量维度可以通过修改 `embedding_dim` 参数调整（默认128维）








