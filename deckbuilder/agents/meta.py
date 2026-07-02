"""Meta agent: drafting and campfire decisions, trained by backprop through V.
Its gradient graph never contains a single card play."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..state import S
from .value import DeckEncoder


class MetaAgent(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.deck_enc = DeckEncoder()
        self.draft = nn.Sequential(nn.Linear(7 + 32 + 1, d), nn.ReLU(),
                                   nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.skip_bias = nn.Parameter(torch.tensor([0.0]))
        self.upgrade = nn.Sequential(nn.Linear(14 + 32 + 1, d), nn.ReLU(),
                                     nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.rest = nn.Sequential(nn.Linear(32 + 1, d), nn.ReLU(), nn.Linear(d, 1))

    def draft_logits(self, offers, counts, hp):
        B = counts.shape[0]
        ctx = torch.cat([self.deck_enc(counts / counts.sum(-1, keepdim=True)),
                         hp.unsqueeze(-1) / 70], -1)
        of = F.one_hot(offers, S.N).float() @ S.FEATS / 10
        x = torch.cat([of, ctx.unsqueeze(1).expand(B, offers.shape[1], ctx.shape[-1])], -1)
        return torch.cat([self.draft(x).squeeze(-1),
                          self.skip_bias.unsqueeze(0).expand(B, 1)], -1)

    def campfire_logits(self, counts, hp):
        B = counts.shape[0]
        ctx = torch.cat([self.deck_enc(counts / counts.sum(-1, keepdim=True)),
                         hp.unsqueeze(-1) / 70], -1)
        pair = torch.cat([S.FEATS[:S.M] / 10, S.FEATS[S.M:] / 10], -1)
        x = torch.cat([pair.unsqueeze(0).expand(B, S.M, 14),
                       ctx.unsqueeze(1).expand(B, S.M, ctx.shape[-1])], -1)
        up = self.upgrade(x).squeeze(-1)
        owned = (counts[:, :S.M] >= 1).float()
        up = up.masked_fill(owned == 0, -1e9)
        return torch.cat([up, self.rest(ctx)], -1)


def apply_draft(w, offers, counts):
    return counts + (w[:, :-1].unsqueeze(-1) * F.one_hot(offers, S.N).float()).sum(1)


def apply_campfire(w, counts, hp, rest_heal, hp_max=70.0):
    delta = w[:, :S.M]
    counts = counts + torch.cat([-delta, delta], -1)
    hp = torch.minimum(hp + w[:, S.M] * rest_heal, torch.full_like(hp, hp_max))
    return counts, hp
