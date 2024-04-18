import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    #input_dim should be input_vocab_size
    def __init__(self, input_dim, output_dim=4, hidden_dim=512, num_heads=8, num_layers=6, dropout_prob=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, 
                                                   nhead=num_heads, 
                                                   dropout=dropout_prob,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        
        # If your transformer encoder layer accepts an attention mask, pass it here
        if attention_mask is not None:
            x = self.transformer_encoder(src=x, src_key_padding_mask=attention_mask)
        else:
            x = self.transformer_encoder(x)

        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
