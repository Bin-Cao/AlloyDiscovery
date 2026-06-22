import torch
import torch.nn as nn


class CompositionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Residual connection
        self.shortcut = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class ProcessingEncoder(nn.Module):
    def __init__(self, cat_dims, cont_dim, emb_dim=16, out_dim=64, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cd + 1, emb_dim) for cd in cat_dims
        ])

        emb_total_dim = emb_dim * len(cat_dims)

        self.cont_net = nn.Sequential(
            nn.Linear(cont_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(emb_total_dim + out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, cat, cont):
        emb = [e(cat[:, i]) for i, e in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=-1)
        cont = self.cont_net(cont)
        combined = torch.cat([emb, cont], dim=-1)
        return self.fusion(combined)


class StateEncoder(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
