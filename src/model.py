import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import CompositionEncoder, ProcessingEncoder, StateEncoder


class AttentionFusion(nn.Module):
    """Attention-based feature fusion"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn = F.softmax(torch.sum(q * k, dim=-1, keepdim=True) / (k.size(-1) ** 0.5), dim=-1)
        return x1 + attn * v


class TrajectoryTabularModel(nn.Module):
    def __init__(self, comp_dim, proc_cat_dims, proc_cont_dim, test_dim,
                 emb_dim=12, proc_out_dim=48, comp_hidden=96, dropout=0.4):
        super().__init__()

        # Encoders with larger capacity
        self.comp_enc = CompositionEncoder(comp_dim, hidden_dim=comp_hidden, dropout=dropout)
        self.proc_enc = ProcessingEncoder(proc_cat_dims, proc_cont_dim,
                                          emb_dim=emb_dim, out_dim=proc_out_dim, dropout=dropout)
        self.state_enc = StateEncoder(in_dim=1, hidden_dim=32, dropout=dropout)

        # Dimensions
        proc_enc_dim = proc_out_dim  # Now returns fixed dim after fusion
        state_enc_dim = 32

        # Attention-based gating mechanism
        self.gate_attn = AttentionFusion(comp_hidden)

        # Enhanced gate network
        self.gate = nn.Sequential(
            nn.Linear(proc_enc_dim + state_enc_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, comp_hidden),
            nn.Sigmoid()
        )

        # Test condition encoder
        self.test_enc = nn.Sequential(
            nn.Linear(test_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Shared feature extraction. Processing/state are included directly;
        # the gate alone is too lossy for small tabular data.
        self.shared = nn.Sequential(
            nn.Linear(comp_hidden + proc_enc_dim + state_enc_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )

        # Multi-task heads (separate for strength and elongation)
        self.strength_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(48, 1)
        )

        self.elongation_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(48, 1)
        )

    def forward(self, batch):
        # Encode all features
        z_comp = self.comp_enc(batch['comp'])
        z_proc = self.proc_enc(batch['proc_cat'], batch['proc_cont'])
        z_state = self.state_enc(batch['state'])
        z_test = self.test_enc(batch['test'])

        # Gating mechanism: modulate composition by processing & state
        gate = self.gate(torch.cat([z_proc, z_state], dim=-1))
        z_comp_gated = gate * z_comp

        # Combine with test conditions
        z = torch.cat([z_comp_gated, z_proc, z_state, z_test], dim=-1)

        # Shared feature extraction
        z_shared = self.shared(z)

        # Multi-task prediction
        strength = self.strength_head(z_shared)
        elongation = self.elongation_head(z_shared)

        return torch.cat([strength, elongation], dim=-1)
