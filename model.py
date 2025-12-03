import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor):
        # x: [B, S, E]
        S = x.size(1)
        return x + self.pe[:S].unsqueeze(0)  # broadcast to [1, S, E]


class TestTransformer(nn.Module):
    def __init__(
        self,
        max_seq_len: int = 1000,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.use_embedding = vocab_size is not None

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(1024, max_seq_len))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor):
        h = self.token_emb(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randint(1, 1000, (8, 827))  # fake float input
    model = TestTransformer(max_seq_len=1000)
    y = model(x)
    print("input:", x.shape)  # torch.Size([8, 827])
    print("output:", y.shape)  # torch.Size([8, 5])
