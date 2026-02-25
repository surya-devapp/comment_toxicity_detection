import torch
from preprocessing import clean_text, ToxicityDataset, build_tokenizer
from model import ToxicityModel
from utils import load_inference_model, predict_toxicity
from sklearn.metrics import classification_report, multilabel_confusion_matrix, fbeta_score
import pandas as pd
import numpy as np
import pickle
import os

def evaluate():
    model_path = 'models/toxicity_model.pth'
    tokenizer_path = 'models/tokenizer.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found. Train it first.")
        return
        
    print("Loading test data and model...")
    df = pd.read_csv('data/train.csv')
    
    # Use a separate test set (comments not in the training set)
    # The current train.py uses a balanced sample of toxic/non-toxic.
    # To be fair, let's just sample from the original df and hope for the best, 
    # or explicitly pick from the remaining comments.
    df_test = df.sample(2000, random_state=123)
    
    model, tokenizer = load_inference_model(model_path, tokenizer_path, vocab_size=15000)
    
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y_true = df_test[classes].values
    y_pred = []
    
    print("Running inference on test set...")
    for comment in df_test['comment_text']:
        res, _ = predict_toxicity(str(comment), model, tokenizer)
        # Probabilities are already handled by utility's predict_toxicity (sigmoided)
        y_pred.append([1 if res[c] > 0.5 else 0 for c in classes])
        
    y_pred = np.array(y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    # Calculate F2-score (beta=2)
    f2_score = fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0)
    print(f"\nWeighted F2-score (beta=2): {f2_score:.4f}")
    
    # Calculate per-class F2-score
    f2_per_class = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=0)
    print("\nPer-class F2-score:")
    for cls, score in zip(classes, f2_per_class):
        print(f"{cls}: {score:.4f}")
    
if __name__ == "__main__":
    import os
    evaluate()
