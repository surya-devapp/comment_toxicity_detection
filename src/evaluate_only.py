import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import load_data 

DATA_PATH = r'D:\ContentMonetizationProj\datasets\projectdatasets\train.csv'
MODEL_PATH = 'models/toxicity_model.pkl'

def evaluate_model():
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if df is None or df.empty:
        print("Data is empty.")
        return

    print(f"Data shape: {df.shape}")

    X = df['clean_text']
    y = df['toxic'].values
    
    # Same split as train.py
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
        
    print("Predicting...")
    try:
        predictions = pipeline.predict(X_test)
        print("Evaluation Report:")
        print(classification_report(y_test, predictions))
    except Exception as e:
         print(f"Error predicting: {e}")

if __name__ == "__main__":
    evaluate_model()
