import argparse
import torch
from torch.utils.data import DataLoader
from dataset import MovieLensDataset, load_data
from model import DSSM
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(args):
    # Load Data
    users, movies, ratings = load_data()
    
    # Dataset
    train_dataset = MovieLensDataset(ratings, users, movies, mode=args.mode, neg_ratio=args.neg_ratio)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = DSSM(
        num_users=train_dataset.num_users,
        num_zips=train_dataset.num_zips,
        num_movies=train_dataset.num_movies,
        num_genres=train_dataset.num_genres,
        embed_dim=args.embed_dim
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss Function
    if args.mode == 'pointwise':
        criterion = nn.MSELoss()
    else:
        criterion = nn.MarginRankingLoss(margin=args.margin)
        
    print(f"Start training in {args.mode} mode...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Move to device
            for k, v in batch.items():
                batch[k] = v.to(args.device)
                
            optimizer.zero_grad()
            
            if args.mode == 'pointwise':
                # Output: scores (B * (1+neg))
                scores = model(batch, mode='pointwise')
                labels = batch['label'].view(-1).to(args.device)
                
                loss = criterion(scores, labels)
                
            elif args.mode == 'pairwise':
                # Output: pos_score, neg_score (B)
                pos_score, neg_score = model(batch, mode='pairwise')
                
                # Target: 1 means pos should be higher than neg
                target = torch.ones_like(pos_score)
                loss = criterion(pos_score, neg_score, target)
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")
        
    # Save Model
    torch.save(model.state_dict(), f"dssm_{args.mode}.pth")
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='pointwise', choices=['pointwise', 'pairwise'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--neg_ratio', type=int, default=3)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    train(args)

