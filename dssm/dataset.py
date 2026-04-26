import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import random
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIRS = [
    os.path.join(REPO_ROOT, "convert_dataset"),
    os.path.join(REPO_ROOT, "converted_dataset"),
]


def resolve_data_dir():
    """定位 convert_dataset.py 生成的统一数据目录。"""
    env_data_dir = os.environ.get("DSSM_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(env_data_dir)

    for data_dir in DEFAULT_DATA_DIRS:
        if os.path.isdir(data_dir):
            return data_dir

    return DEFAULT_DATA_DIRS[0]


DATA_DIR = resolve_data_dir()

class MovieLensDataset(Dataset):
    def __init__(self, ratings, users, movies, mode='pointwise', neg_ratio=3, margin=0.2):
        """
        Args:
            ratings: DataFrame, 评分数据
            users: DataFrame, 用户数据
            movies: DataFrame, 电影数据
            mode: str, 'pointwise' or 'pairwise'
            neg_ratio: int, Pointwise 模式下的正负样本比例 1:neg_ratio
            margin: float, Pairwise loss margin (虽然 dataset 不直接用 margin，但可能影响采样逻辑)
        """
        self.mode = mode
        self.neg_ratio = neg_ratio
        
        # 预处理数据
        self.users, self.movies, self.ratings = self._preprocess(users, movies, ratings)
        
        # 计算物品热门程度用于负采样
        self.movie_pop = self.ratings['MovieID'].value_counts()
        self.all_movie_ids = self.movie_pop.index.values
        # 抽样概率与点击次数的 0.75 次方成正比
        self.pop_probs = np.power(self.movie_pop.values, 0.75)
        self.pop_probs = self.pop_probs / self.pop_probs.sum()
        
        # 生成样本列表
        self.samples = self._generate_samples()
        
    def _preprocess(self, users, movies, ratings):
        # 编码 UserID
        self.user_encoder = LabelEncoder()
        users['UserID_idx'] = self.user_encoder.fit_transform(users['UserID'])
        self.num_users = len(self.user_encoder.classes_)
        
        # 统一转换后的数据可能来自不同 MovieLens 版本，分类特征统一用 LabelEncoder。
        self.gender_encoder = LabelEncoder()
        users['Gender_idx'] = self.gender_encoder.fit_transform(users['Gender'].astype(str))
        self.num_genders = len(self.gender_encoder.classes_)

        self.age_encoder = LabelEncoder()
        users['Age_idx'] = self.age_encoder.fit_transform(users['Age'].astype(str))
        self.num_ages = len(self.age_encoder.classes_)

        self.occupation_encoder = LabelEncoder()
        users['Occupation_idx'] = self.occupation_encoder.fit_transform(users['Occupation'].astype(str))
        self.num_occupations = len(self.occupation_encoder.classes_)
        
        # 编码 Zip-code
        self.zip_encoder = LabelEncoder()
        users['Zip_idx'] = self.zip_encoder.fit_transform(users['Zip-code'])
        self.num_zips = len(self.zip_encoder.classes_)
        
        # 编码 MovieID
        self.movie_encoder = LabelEncoder()
        movies['MovieID_idx'] = self.movie_encoder.fit_transform(movies['MovieID'])
        self.num_movies = len(self.movie_encoder.classes_)
        
        # 处理 Genres (Multi-hot -> Padding/Truncating)
        # 这里为了简化，我们只取第一个 Genre 或者用 Multi-hot EmbeddingBag
        # 为了兼容上面的要求 "用两个embedding层把MovieID和Genres映射到向量"，我们假设 Genre 是变长的
        # 先建立 Genre 词表
        all_genres = set()
        for g in movies['Genres']:
            all_genres.update(g.split('|'))
        self.genre_encoder = {g: i+1 for i, g in enumerate(all_genres)} # 0 for padding
        self.num_genres = len(self.genre_encoder) + 1
        
        # 将 Movie 的 Genres 转换为定长列表 (e.g. max len 5)
        max_genres = 5
        movie_genre_indices = []
        for g in movies['Genres']:
            indices = [self.genre_encoder[x] for x in g.split('|')][:max_genres]
            indices += [0] * (max_genres - len(indices))
            movie_genre_indices.append(indices)
        movies['Genres_idx'] = movie_genre_indices
        
        # 合并 User 和 Movie 信息到 Ratings
        ratings = ratings.merge(users[['UserID', 'UserID_idx', 'Gender_idx', 'Age_idx', 'Occupation_idx', 'Zip_idx']], on='UserID')
        ratings = ratings.merge(movies[['MovieID', 'MovieID_idx']], on='MovieID')
        
        # 存储特征查找表以便快速获取
        self.user_features = users.set_index('UserID_idx')[['Gender_idx', 'Age_idx', 'Occupation_idx', 'Zip_idx']].to_dict('index')
        self.movie_features = movies.set_index('MovieID_idx')[['Genres_idx']].to_dict('index')
        
        # 仅保留正样本 (Rating >= 4) 作为交互
        # 或者保留所有交互作为正样本 (隐式反馈假设)
        # 这里假设所有评分过的都是正样本，但通常会过滤低分
        # 为了简单，我们只保留 Rating >= 3 的作为正样本
        pos_ratings = ratings[ratings['Rating'] >= 3].copy()
        
        return users, movies, pos_ratings

    def _generate_samples(self):
        """生成训练样本 (User, Item) 对"""
        return self.ratings[['UserID_idx', 'MovieID_idx']].values.tolist()

    def _sample_negatives(self, count=1):
        """根据热门程度抽样负样本"""
        # np.random.choice 比较慢，可以用 alias method 优化，这里简单处理
        return np.random.choice(self.all_movie_ids, size=count, p=self.pop_probs)
        # 注意: 采出来的 MovieID 需要转为 MovieID_idx
        # 由于 self.all_movie_ids 是原始 ID，需要转换
        # 优化: self.pop_probs 对应的 index 应该是 MovieID_idx
        # 重新计算 pop_probs 基于 MovieID_idx
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, pos_movie_idx = self.samples[idx]
        
        # 获取用户特征
        u_feat = self.user_features[user_idx]
        
        # 获取正样本电影特征
        m_pos_feat = self.movie_features[pos_movie_idx]
        
        if self.mode == 'pointwise':
            # Pointwise: 1个正样本 + neg_ratio 个负样本
            # 实际上 Dataset 最好每次返回一条 (User, Item, Label)
            # 但为了效率，我们可以在这里随机决定是返回正样本还是负样本
            # 或者，更常见的是：Dataset 只包含正样本，在 Collate Fn 中采样负样本
            # 这里我们在 __getitem__ 动态返回 1 个正样本和 N 个负样本
            
            # 正样本
            items = [pos_movie_idx]
            labels = [1.0]
            
            # 负样本
            neg_indices = np.random.randint(0, self.num_movies, size=self.neg_ratio)
            items.extend(neg_indices)
            labels.extend([-1.0] * self.neg_ratio)
            
            return {
                'user_id': user_idx,
                'gender': u_feat['Gender_idx'],
                'age': u_feat['Age_idx'],
                'occupation': u_feat['Occupation_idx'],
                'zip': u_feat['Zip_idx'],
                'movie_id': np.array(items), # (1 + neg_ratio)
                'genres': np.array([self.movie_features[m]['Genres_idx'] for m in items]), # (1 + neg_ratio, 5)
                'label': np.array(labels, dtype=np.float32)
            }
            
        elif self.mode == 'pairwise':
            # Pairwise: User, Pos Item, Neg Item
            neg_idx = np.random.randint(0, self.num_movies) # 简化为 uniform，后续优化
            m_neg_feat = self.movie_features[neg_idx]
            
            return {
                'user_id': user_idx,
                'gender': u_feat['Gender_idx'],
                'age': u_feat['Age_idx'],
                'occupation': u_feat['Occupation_idx'],
                'zip': u_feat['Zip_idx'],
                'pos_movie_id': pos_movie_idx,
                'pos_genres': np.array(m_pos_feat['Genres_idx']),
                'neg_movie_id': neg_idx,
                'neg_genres': np.array(m_neg_feat['Genres_idx'])
            }

def load_data(ratings_filename="ratings.csv"):
    """加载 convert_dataset.py 输出的统一 CSV 数据。"""
    print(f"Loading converted data from {DATA_DIR}...")
    users_path = os.path.join(DATA_DIR, "users.csv")
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, ratings_filename)

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
    ratings = pd.read_csv(ratings_path).rename(
        columns={
            "user_id": "UserID",
            "movie_id": "MovieID",
            "rating": "Rating",
            "timestamp": "Timestamp",
        }
    )

    users = users[["UserID", "Gender", "Age", "Occupation", "Zip-code"]].copy()
    movies = movies[["MovieID", "Title", "Genres"]].copy()
    ratings = ratings[["UserID", "MovieID", "Rating", "Timestamp"]].copy()
    users["Gender"] = users["Gender"].fillna("Unknown").astype(str)
    users["Age"] = users["Age"].fillna(0)
    users["Occupation"] = users["Occupation"].fillna("Unknown").astype(str)
    users["Zip-code"] = users["Zip-code"].fillna("").astype(str)
    movies["Genres"] = movies["Genres"].fillna("(no genres listed)").astype(str)
    return users, movies, ratings

