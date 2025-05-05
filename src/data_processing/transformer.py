import pandas as pd
import torch.nn as nn
from sympy.printing.pytorch import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear


class Transformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3, pred_steps=5):
        super().__init__()
        # self.input_proj = Linear(input_dim, 64)  # Project multiple features
        # self.transformer = TransformerEncoderLayer(64, nhead=4)
        # self.output_proj = Linear(64, output_dim)
        self.pred_steps = pred_steps
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.projection = nn.Linear(d_model, pred_steps)

    def forward(self, x):
        # if isinstance(x, pd.DataFrame):
        #     x = torch.tensor(x.values, dtype=torch.float32)
        # x = self.input_proj(x)
        # x = self.transformer(x)
        # return self.output_proj(x)
        x = x.unsqueeze(-1)  # [seq_len, batch_size, 1]
        x = self.embedding(x)  # [seq_len, batch_size, d_model]
        x = self.transformer(x)  # [seq_len, batch_size, d_model]
        return self.projection(x.mean(0))  # [batch_size, pred_steps]