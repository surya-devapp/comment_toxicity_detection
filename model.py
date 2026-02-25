import torch
import torch.nn as nn

class ToxicityModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128):
        super(ToxicityModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 6)
        
    def forward(self, x):
        # x: [batch_size, sequence_length]
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Global Max Pooling
        pooled, _ = torch.max(lstm_out, dim=1)
        
        x = torch.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x # Returns Logits for BCEWithLogitsLoss
