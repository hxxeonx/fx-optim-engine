import torch
import torch.nn as nn
import math



class DataEmbedding(nn.Module):
    def __init__(self, t_interval, n_vars, d_model, pe_method='zero', learn_pe=True, dropout=0.1):
        super(DataEmbedding, self).__init__()

        q_len           = t_interval
        self.seq_len    = q_len
        self.dropout    = nn.Dropout(dropout)

        self.I_emb      = nn.Linear(n_vars, d_model)                                        
        self.W_pos      = self.positional_encoding(pe_method, learn_pe, q_len, d_model)

    def positional_encoding(self, pe_method, learn_pe, q_len, d_model):

        if pe_method == None:
            W_pos = torch.empty((q_len, d_model)) 
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False

        elif pe_method == 'sincos': W_pos = self.sinusoidal_encoding(q_len, d_model, normalize=True)

        else: raise ValueError(f"{pe_method} is not a valid pe (positional encoder. Available types: 'sincos', None.)")
        
        return nn.Parameter(W_pos, requires_grad = learn_pe)

    def sinusoidal_encoding(self, q_len, d_model, normalize=True):
        pe = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if normalize:
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)
        return pe

    def forward(self, x):                                                  

        x = self.I_emb(x)                                                            # Input Embedding     - x: [bs x n_vars x time_interval x d_model]
        u = self.dropout(x + self.W_pos)                                             # Summation (IE + PE) - u: [bs * nvars x time_interval x d_model]
    
        return u