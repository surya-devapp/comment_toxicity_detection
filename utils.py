import torch
import torch.nn as nn
from model import ToxicityModel
from preprocessing import clean_text
import pickle
import numpy as np
import re
try:
    from langdetect import detect
except ImportError:
    detect = None

def load_inference_model(model_path, tokenizer_path, vocab_size=15000):
    model = ToxicityModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    return model, tokenizer

def detect_language(text):
    if not detect:
        return "unknown"
    if not text or str(text).strip() == "":
        return "empty"
    try:
        return detect(text)
    except:
        return "error"

def predict_toxicity(text, model, tokenizer, max_len=200, mask_entities=False):
    cleaned = clean_text(text, mask_entities=mask_entities)
    tokens = cleaned.split()
    token_ids = [tokenizer.get(token, 1) for token in tokens[:max_len]]
    
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Apply sigmoid because model now returns logits
        probs = torch.sigmoid(output).squeeze(0).numpy()
    
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    return dict(zip(classes, probs)), cleaned
def get_toxicity_status(probs, safe_threshold=0.4, toxic_threshold=0.7):
    max_prob = max(probs.values())
    
    # Midpoint decision for binary classification
    # If the score is in the neutral zone, we assign a label but add the 'Human Review' flag
    is_edge_case = safe_threshold <= max_prob < toxic_threshold
    review_suffix = " (Needs Human Review)" if is_edge_case else ""
    
    if max_prob < 0.5:
        return f"Safe{review_suffix}", "green" if not is_edge_case else "orange"
    else:
        return f"Toxic{review_suffix}", "red" if not is_edge_case else "orange"
