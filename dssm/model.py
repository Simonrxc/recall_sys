import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, num_zips, embed_dim=32, hidden_dims=[256, 128, 64]):
        super().__init__()
        # Embeddings
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.gender_emb = nn.Embedding(2, embed_dim) # 0/1
        self.age_emb = nn.Embedding(7, embed_dim)    # 7 buckets
        self.occ_emb = nn.Embedding(21, embed_dim)   # 21 occupations
        self.zip_emb = nn.Embedding(num_zips, embed_dim)
        
        # Concat dim = 5 * embed_dim
        input_dim = 5 * embed_dim
        
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]
        
    def forward(self, uid, gender, age, occ, zip_code):
        u = self.user_emb(uid)
        g = self.gender_emb(gender)
        a = self.age_emb(age)
        o = self.occ_emb(occ)
        z = self.zip_emb(zip_code)
        
        # Concat
        x = torch.cat([u, g, a, o, z], dim=1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1) # Normalize for Cosine

class ItemTower(nn.Module):
    def __init__(self, num_movies, num_genres, embed_dim=32, hidden_dims=[256, 128, 64]):
        super().__init__()
        self.movie_emb = nn.Embedding(num_movies, embed_dim)
        # Genres: padding_idx=0
        self.genre_emb = nn.Embedding(num_genres, embed_dim, padding_idx=0)
        
        # Concat dim = 2 * embed_dim
        input_dim = 2 * embed_dim
        
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]
        
    def forward(self, mid, genres):
        # mid: (B)
        # genres: (B, 5)
        m = self.movie_emb(mid)
        
        # Average/Sum pooling for genres
        # genres_emb: (B, 5, D) -> (B, D)
        g = self.genre_emb(genres).mean(dim=1)
        
        x = torch.cat([m, g], dim=1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)

class DSSM(nn.Module):
    def __init__(self, num_users, num_zips, num_movies, num_genres, embed_dim=32):
        super().__init__()
        self.user_tower = UserTower(num_users, num_zips, embed_dim)
        self.item_tower = ItemTower(num_movies, num_genres, embed_dim)
        
    def forward(self, batch_data, mode='pointwise'):
        # User side
        u_vec = self.user_tower(
            batch_data['user_id'],
            batch_data['gender'],
            batch_data['age'],
            batch_data['occupation'],
            batch_data['zip']
        )
        
        if mode == 'pointwise':
            # batch_data['movie_id'] shape: (B, 1+neg)
            # Flatten to process
            B, N = batch_data['movie_id'].shape
            flat_mids = batch_data['movie_id'].view(-1)
            flat_genres = batch_data['genres'].view(-1, 5)
            
            i_vec = self.item_tower(flat_mids, flat_genres)
            
            # u_vec: (B, D) -> (B, 1, D) -> (B, 1+neg, D) -> flatten -> (B*(1+neg), D)
            u_vec_exp = u_vec.unsqueeze(1).expand(-1, N, -1).reshape(-1, u_vec.shape[-1])
            
            # Cosine similarity (already normalized)
            # (B*(1+neg), D) * (B*(1+neg), D) -> sum -> (B*(1+neg))
            scores = (u_vec_exp * i_vec).sum(dim=1)
            return scores
            
        elif mode == 'pairwise':
            # Pos Item
            pos_mids = batch_data['pos_movie_id']
            pos_genres = batch_data['pos_genres']
            pos_vec = self.item_tower(pos_mids, pos_genres)
            
            # Neg Item
            neg_mids = batch_data['neg_movie_id']
            neg_genres = batch_data['neg_genres']
            neg_vec = self.item_tower(neg_mids, neg_genres)
            
            # Scores
            pos_score = (u_vec * pos_vec).sum(dim=1)
            neg_score = (u_vec * neg_vec).sum(dim=1)
            
            return pos_score, neg_score

