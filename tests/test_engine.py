"""Regression tests encoding this project's war stories. If one of these
fails, a previously-caught bug is back."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from deckbuilder import S, combat, sample_enemy_conditions, RandomPolicy
from deckbuilder.engine.combat import pick_intents

torch.manual_seed(0)


class CheatPolicy(nn.Module):
    """Outputs enormous scores for every card. Under a SOFT mask this policy
    could play cards not in its hand (the v2 mask exploit)."""
    def __init__(self):
        super().__init__()
        self.pass_bias = nn.Parameter(torch.tensor([-100.0]))
    def forward(self, x):
        return torch.full(x.shape[:-1] + (1,), 100.0)


def _all_defend(B):
    d = F.one_hot(torch.tensor(S.NAMES.index("Defend")), S.N).float()
    return d.unsqueeze(0).expand(B, S.N).contiguous()


def test_hard_mask_blocks_unowned_cards():
    """THE mask-exploit regression: an all-Defend deck deals zero damage, so
    the enemy must survive at full HP no matter how badly the policy wants
    to play attacks it does not hold."""
    B = 256
    etype, hp0, ats = sample_enemy_conditions(B, hp_scale=1.5)
    out = combat(CheatPolicy(), _all_defend(B), etype, hp0, ats, torch.full((B,), 60.0))
    assert out["won"].sum() == 0


def test_all_defend_cannot_win():
    B = 512
    etype, hp0, ats = sample_enemy_conditions(B, hp_scale=1.5)
    out = combat(RandomPolicy(), _all_defend(B), etype, hp0, ats, torch.full((B,), 60.0))
    assert out["won"].sum() == 0


def test_deck_quality_matters():
    """Deck-sensitivity is the falsifiable prediction of the whole design."""
    B = 2000
    strong = torch.zeros(S.N)
    for n, c in [("Ember Slash", 4), ("Heavy Blade", 3), ("Inflame", 3)]:
        strong[S.NAMES.index(n)] = c
    strong = (strong / strong.sum()).unsqueeze(0).expand(B, S.N).contiguous()
    starter = (S.STARTER / S.STARTER.sum()).unsqueeze(0).expand(B, S.N).contiguous()
    torch.manual_seed(1)
    etype, hp0, ats = sample_enemy_conditions(B, hp_scale=1.6)
    w_strong = combat(RandomPolicy(), strong, etype, hp0, ats, torch.full((B,), 60.0))["won"].mean()
    torch.manual_seed(1)
    etype, hp0, ats = sample_enemy_conditions(B, hp_scale=1.6)
    w_start = combat(RandomPolicy(), starter, etype, hp0, ats, torch.full((B,), 60.0))["won"].mean()
    assert w_strong > w_start + 0.05


def test_cycle_pattern_cycles():
    cultist = torch.full((16,), S.ENAMES.index("Cultist"), dtype=torch.long)
    last = torch.zeros(16, dtype=torch.long); rep = torch.zeros(16, dtype=torch.long)
    seq = []
    for t in range(6):
        idx, rep = pick_intents(cultist, t, last, rep)
        last = idx
        seq.append(idx[0].item())
    assert seq == [0, 1, 0, 1, 0, 1]


def test_weighted_pattern_respects_max_repeat():
    jw = torch.full((64,), S.ENAMES.index("Jaw Worm"), dtype=torch.long)
    last = torch.zeros(64, dtype=torch.long); rep = torch.zeros(64, dtype=torch.long)
    seqs = [[] for _ in range(64)]
    for t in range(60):
        idx, rep = pick_intents(jw, t, last, rep)
        last = idx
        for b in range(64):
            seqs[b].append(idx[b].item())
    for s in seqs:
        run_len = 1
        for a, b in zip(s, s[1:]):
            run_len = run_len + 1 if a == b else 1
            assert run_len <= 3   # max_repeat=2 -> never unboundedly repeated
        assert len(set(s)) > 1    # uses more than one intent


def test_upgrades_are_strictly_better_rows():
    base, up = S.FEATS[:S.M], S.FEATS[S.M:]
    assert (up[:, 1:] >= base[:, 1:]).all()          # never worse stats
    assert (up[:, 1:] > base[:, 1:]).any(dim=1).all()  # each upgrade improves something
    assert (up[:, 0] == base[:, 0]).all()            # cost unchanged
