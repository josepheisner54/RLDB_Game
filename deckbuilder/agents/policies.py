"""Combat policies. Registry pattern: add architectures here and register them.

ARCHITECTURE STATUS (contribution seam):
- MLPCombatPolicy: weight-shared per-card scorer. No cross-card interaction --
  scoring card i never sees the rest of the hand; hand synergy is only
  representable diffusely through state scalars across sequential plays.
- AttentionCombatPolicy: STUB. Intended design: hand cards as tokens
  (feature-embedded), plus a state token and an enemy token; 2 layers of
  self-attention; per-token score head; pass scored from the state token.
  Slots into combat() unchanged: forward(inp) with inp (B, N, 21) must
  return (B, N, 1) scores, plus a .pass_bias parameter.
"""
import torch
import torch.nn as nn


class MLPCombatPolicy(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14 + 7, d), nn.ReLU(),
                                 nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.pass_bias = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, x):
        return self.net(x)


class RandomPolicy(nn.Module):
    """Uniform over playable cards -- the honest baseline for every eval."""
    def __init__(self):
        super().__init__()
        self.pass_bias = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, x):
        return torch.zeros(x.shape[:-1] + (1,), device=x.device)


class AttentionCombatPolicy(nn.Module):
    """TODO(collaborator): see module docstring for the intended design."""
    def __init__(self, d=96, n_layers=2, n_heads=4):
        super().__init__()
        raise NotImplementedError("The seam is yours -- see docstring.")


POLICIES = {"mlp": MLPCombatPolicy, "random": RandomPolicy, "attention": AttentionCombatPolicy}


def make_policy(name="mlp", **kw):
    return POLICIES[name](**kw)
