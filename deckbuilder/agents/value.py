"""Value function: the interface between combat and meta agents.
Earns its keep twice: (1) condition-aware actor-critic baseline,
(2) the meta agent's differentiable world model.

DeckEncoder is a contribution seam: currently one linear layer on the
count vector (DeepSets with identity phi). Intended upgrade: attention
pooling over (card_feature_embedding, count) pairs -- captures composition
interactions and becomes card-pool-agnostic (no fixed input width).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..state import S


class DeckEncoder(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.proj = nn.Linear(S.N, d)

    def forward(self, deck_probs):
        return F.relu(self.proj(deck_probs))


class ValueNet(nn.Module):
    def __init__(self, d=128, deck_encoder=None):
        super().__init__()
        self.deck_enc = deck_encoder or DeckEncoder()
        self.net = nn.Sequential(nn.Linear(1 + 4 + 32, d), nn.ReLU(),
                                 nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, hp, esum, deck_probs):
        x = torch.cat([hp.unsqueeze(-1) / 70, esum, self.deck_enc(deck_probs)], -1)
        out = self.net(x)
        return out[:, 0] * 70, torch.sigmoid(out[:, 1])
