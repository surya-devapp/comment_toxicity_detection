import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import clean_text, build_tokenizer, ToxicityDataset
from model import ToxicityModel
from sklearn.model_selection import train_test_split
import pickle

def train():
    # Load dataset
    print("Loading data...")
    df = pd.read_csv('data/train.csv')
    
    # 1. Handle missing/empty/error data
    df = df.dropna(subset=['comment_text'])
    # Filter out empty strings and common error placeholders like #ERROR!
    df = df[df['comment_text'].astype(str).str.strip() != ""]
    df = df[~df['comment_text'].astype(str).str.contains("#ERROR!", na=False)]
    
    # 2. Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['comment_text'])
    dropped_duplicates = initial_len - len(df)
    if dropped_duplicates > 0:
        print(f"Removed {dropped_duplicates} duplicate comments from dataset.")
    
    # 3. Balanced sampling strategy:
    # Get all toxic comments (any label is 1)
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_toxic = df[df[classes].sum(axis=1) > 0]
    
    # Adjusted sampling: focus more on toxicity
    # Use clean sample 1.2x of toxic (instead of equal) to keep context but favor sensitivity
    df_non_toxic = df[df[classes].sum(axis=1) == 0].sample(int(len(df_toxic) * 1.2), random_state=42)
    
    # 3. Combine and shuffle
    df_balanced = pd.concat([df_toxic, df_non_toxic]).sample(frac=1, random_state=42)
    
    print(f"Balanced Dataset Size: {len(df_balanced)} (Toxic: {len(df_toxic)}, Non-Toxic: {len(df_non_toxic)})")
    
    print("Cleaning text...")
    df_balanced['comment_text'] = df_balanced['comment_text'].apply(clean_text)
    
    print("Building tokenizer...")
    texts = df_balanced['comment_text'].values
    labels = df_balanced[classes].values
    
    tokenizer = build_tokenizer(texts, max_features=15000) # Increased features
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    # Create Datasets and Dataloaders
    train_ds = ToxicityDataset(X_train, y_train, tokenizer)
    test_ds = ToxicityDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    print("Building model...")
    model = ToxicityModel(vocab_size=15000, hidden_dim=128)
    
    # Weighted Loss: Penalize false negatives for minority classes
    # Weights for [toxic, severe_toxic, obscene, threat, insult, identity_hate]
    pos_weights = torch.tensor([2.0, 4.0, 2.0, 5.0, 2.0, 4.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower learning rate
    
    epochs = 6
    print(f"Training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}: Loss {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss/len(test_loader)
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}, val loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), 'models/toxicity_model.pth')

    print("Starting Adversarial Fine-tuning (Phase 3)...")
    if os.path.exists('data/correction_set.csv'):
        df_corr = pd.read_csv('data/correction_set.csv')
        df_corr['comment_text'] = df_corr['comment_text'].apply(clean_text)
        
        corr_ds = ToxicityDataset(df_corr['comment_text'].values, df_corr[classes].values, tokenizer)
        corr_loader = DataLoader(corr_ds, batch_size=len(df_corr)) # Process as one batch
        
        # Fine-tune with a very low learning rate and high sample weight simulation
        # Since standard BCE doesn't take sample weights easily per batch in this setup, 
        # we repeat the correction samples or use a custom loss. 
        # For simplicity, we just run a few targeted iterations.
        model.train()
        fine_optimizer = optim.Adam(model.parameters(), lr=1e-5) 
        
        for i in range(20): # 20 iterations on the correction set
            for data, target in corr_loader:
                fine_optimizer.zero_grad()
                output = model(data)
                # We want the model to be VERY sure these are safe (target is all 0s)
                loss = nn.BCEWithLogitsLoss()(output, target) * 5.0 # Weight these 5x
                loss.backward()
                fine_optimizer.step()
            
            if i % 5 == 0:
                print(f"Fine-tuning Iteration {i}: Loss {loss.item():.4f}")
        
        print("Adversarial Fine-tuning complete.")
        # Save the fine-tuned model
        torch.save(model.state_dict(), 'models/toxicity_model.pth')
    else:
        print("Correction set not found at data/correction_set.csv. Skipping fine-tuning.")

    print("Saving tokenizer...")
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Training complete.")

if __name__ == "__main__":
    train()
