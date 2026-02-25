import pandas as pd
import numpy as np
import re
import string
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

def clean_text(text, mask_entities=False):
    if mask_entities:
        # Heuristic: Mask capitalized words not at the beginning of a sentence
        text = re.sub(r'(?<!^)(?<!\. )(?<!\? )(?<!\! )\b[A-Z][a-z]+\b', '[ENTITY]', text)
        
    text = str(text).lower()
    # Remove all [brackets] except [entity]
    text = re.sub(r'\[(?!(?:entity)\]).*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    # Remove punctuation but keep our brackets for [entity]
    punc_to_remove = string.punctuation.replace('[','').replace(']','')
    text = re.sub(r'[%s]' % re.escape(punc_to_remove), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Simple tokenization
        tokens = text.split()
        token_ids = [self.tokenizer.get(token, 1) for token in tokens[:self.max_len]]
        
        # Padding
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def build_tokenizer(texts, max_features=10000):
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    tokenizer = {word: i+2 for i, (word, count) in enumerate(sorted_words[:max_features-2])}
    tokenizer['<PAD>'] = 0
    tokenizer['<OOV>'] = 1
    return tokenizer

if __name__ == "__main__":
    test_text = "This is a clean text example."
    print(clean_text(test_text))
