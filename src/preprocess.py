import re
import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100

class SimpleTokenizer:
    def __init__(self, num_words=MAX_NUM_WORDS):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        all_text = " ".join([str(t) for t in texts])
        words = all_text.split()
        count = Counter(words)
        # Sort by most common
        most_common = count.most_common(self.num_words - 1) # -1 for padding/unknown
        
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        for idx, (word, _) in enumerate(most_common, 2):
            self.word_index[word] = idx
        
        self.index_word = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = str(text).split()
            seq = [self.word_index.get(w, self.word_index["<UNK>"]) for w in words]
            sequences.append(seq)
        return sequences

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Generating dummy data...")
        data = {
            'comment_text': [
                "This is an amazing post!", 
                "You are an idiot and I hate you.", 
                "Have a great day everyone.", 
                "Stop checking this, it's stupid.",
                "I love this community.",
                "You are the worst person ever."
            ] * 100,
            'toxic': [0, 1, 0, 1, 0, 1] * 100
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    
    if 'comment_text' not in df.columns or 'toxic' not in df.columns:
        pass
        
    df['clean_text'] = df['comment_text'].apply(clean_text)
    return df

def prepare_tokenizer(texts, save_path='models/tokenizer.pickle'):
    tokenizer = SimpleTokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer

def load_tokenizer(path='models/tokenizer.pickle'):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def pad_sequences(sequences, maxlen, padding='post', value=0):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded.append(seq[:maxlen])
        else:
            if padding == 'post':
                padded.append(seq + [value] * (maxlen - len(seq)))
            else:
                padded.append([value] * (maxlen - len(seq)) + seq)
    return np.array(padded)

def preprocess_input(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data
