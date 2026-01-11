import torch
import torch.nn as nn

class ToxicityModel(nn.Module):
    def __init__(self, num_words, embed_dim=128, hidden_dim=64):
        super(ToxicityModel, self).__init__()
        self.embedding = nn.Embedding(num_words, embed_dim)
        # Bidirectional LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer
        # Hidden dim * 2 because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # lstm returns: output, (h_n, c_n)
        # we generally take the last hidden state or pool the outputs
        # output shape: (batch, seq_len, hidden*2)
        lstm_out, _ = self.lstm(x)
        
        # Simple Global Max Pooling or take last step
        # Let's take the mean of the sequence outputs
        x = torch.mean(lstm_out, dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
