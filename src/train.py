import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import pickle
from sklearn.calibration import CalibratedClassifierCV
from features import SentimentExtractor, TargetExtractor
from preprocess import load_data, clean_text

DATA_PATH = r'D:\ContentMonetizationProj\datasets\projectdatasets\train.csv'
MODEL_PATH = 'models/toxicity_model_v3_enhanced.pkl' 

def train():
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    # Use generic generic cleaning from preprocess
    # We will use TfidfVectorizer inside the pipeline for end-to-end simplicity
    
    X = df['clean_text']
    y = df['toxic'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Define a Neural Network Pipeline
    # MLPClassifier is a Multi-layer Perceptron (Deep Neural Network)
    # Wrapped in CalibratedClassifierCV for better probability estimates
    print("Creating Neural Network Pipeline with Feature Engineering...")
    
    # Feature Union combines text vectors with sentiment and target detection
    combined_features = FeatureUnion([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('sentiment', SentimentExtractor()),
        ('target_detect', TargetExtractor())
    ])
    
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=300, random_state=42, verbose=True)
    calibrated_mlp = CalibratedClassifierCV(mlp, method='sigmoid', cv=3)

    pipeline = Pipeline([
        ('features', combined_features),
        ('mlp', calibrated_mlp) 
    ])
    
    print("Training model (this may take a moment)...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train()
