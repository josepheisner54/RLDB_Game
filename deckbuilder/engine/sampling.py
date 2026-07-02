"""Condition generators. Lessons encoded here:
- decks are bounded deviations from the starter (the reachable set), incl. upgrades
- HP ranges reach low (coverage: campfires visit low-HP states)
- stat jitter is observable via featurization (and pre-trains upgrade piloting)
"""
import torch
import torch.nn.functional as F
from ..state import S, DEVICE, DMG, BLK


def sample_enemy_conditions(B, pool=None, hp_scale=None, atk_scale=None):
    if pool is None:
        etype = torch.randint(0, S.E, (B,), device=DEVICE)
    else:
        ids = torch.tensor([S.ENAMES.index(n) for n in pool], device=DEVICE)
        etype = ids[torch.randint(0, len(ids), (B,), device=DEVICE)]
    lo, hi = S.E_HP[etype, 0], S.E_HP[etype, 1]
    hs = (torch.rand(B, device=DEVICE) * 1.5 + 0.9) if hp_scale is None else torch.full((B,), hp_scale, device=DEVICE)
    ats = (torch.rand(B, device=DEVICE) * 0.55 + 0.85) if atk_scale is None else torch.full((B,), atk_scale, device=DEVICE)
    hp0 = (torch.rand(B, device=DEVICE) * (hi - lo) + lo) * hs
    return etype, hp0, ats


def fight_conditions(B, fight):
    f = S.FIGHTS[fight]
    return sample_enemy_conditions(B, pool=f["pool"], hp_scale=f["hp_scale"], atk_scale=f["atk_scale"])


def enemy_summary(etype, hp0, atk_scale):
    s = S.E_SUMMARY[etype]
    return torch.stack([hp0 / 100, s[:, 0] * atk_scale / 14, s[:, 1] / 6, s[:, 2] / 3], -1)


def sample_decks(B, max_drafts=4, broad_frac=0.1, upgrade_prob=0.5):
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    k = torch.randint(0, max_drafts + 1, (B,), device=DEVICE)
    deep = torch.randint(0, 9, (B,), device=DEVICE)
    k = torch.where(torch.rand(B, device=DEVICE) < broad_frac, deep, k)
    for i in range(8):
        add = F.one_hot(torch.randint(0, S.M, (B,), device=DEVICE), S.N).float()
        counts = counts + add * (k > i).float().unsqueeze(-1)
    for _ in range(2):
        do = (torch.rand(B, device=DEVICE) < upgrade_prob).float()
        base = counts[:, :S.M]
        pick = torch.multinomial(base.clamp(min=1e-8), 1).squeeze(-1)
        has = (base.gather(1, pick.unsqueeze(1)).squeeze(1) >= 1).float() * do
        delta = F.one_hot(pick, S.M).float() * has.unsqueeze(-1)
        counts = counts + torch.cat([-delta, delta], -1)
    return counts / counts.sum(-1, keepdim=True)


def jitter_feats(B, amount=1.0):
    f = S.FEATS.unsqueeze(0).repeat(B, 1, 1)
    if amount > 0:
        noise = (torch.rand(B, S.N, 2, device=DEVICE) * 2 - 1) * amount
        mask = (f[:, :, [DMG, BLK]] > 0).float()
        f[:, :, [DMG, BLK]] = (f[:, :, [DMG, BLK]] + noise * mask).clamp(min=0)
    return f
