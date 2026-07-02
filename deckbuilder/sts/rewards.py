"""Card rewards: 3 offers per combat, rarity by drop table + pity offset.

Drop tables (percent):        common  uncommon  rare
    normal combat               60       37       3
    elite combat                50       40      10
    boss                         0        0     100

Pity ("card blizzard"): rare_offset is per-run state, added to the rare
percentage. A common roll bumps it +1 (cap 40); a rare roll resets it to -5.
Within one 3-card offer, duplicates are excluded.
"""
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TABLES = {  # (common, uncommon, rare) base percents
    "normal": (60.0, 37.0, 3.0),
    "elite": (50.0, 40.0, 10.0),
    "boss": (0.0, 0.0, 100.0),
}
N_OFFERS = 3
PITY_START, PITY_CAP, PITY_RESET = -5.0, 40.0, -5.0


def init_pity(B):
    return torch.full((B,), PITY_START, device=DEVICE)


def roll_offers(C, B, source, rare_offset):
    """Returns offers (B, N_OFFERS) base-card rows and the updated pity.
    `source` in {"normal","elite","boss"}."""
    c0, u0, r0 = TABLES[source]
    pool_mask = torch.zeros(3, C.M, device=DEVICE)
    for ri, rn in enumerate(("common", "uncommon", "rare")):
        pool_mask[ri, C.POOL[rn]] = 1.0
    offers = torch.full((B, N_OFFERS), -1, dtype=torch.long, device=DEVICE)
    taken = torch.zeros(B, C.M, device=DEVICE)
    for k in range(N_OFFERS):
        r = (r0 + rare_offset).clamp(min=0.0, max=100.0)
        u = torch.full_like(r, u0)
        c = (100.0 - r - u).clamp(min=0.0)
        if source == "boss":
            r, u, c = torch.full_like(r, 100.0), u * 0, c * 0
        rarity = torch.multinomial(torch.stack([c, u, r], -1).clamp(min=1e-9), 1).squeeze(-1)
        # pity updates
        rare_offset = torch.where(rarity == 0, (rare_offset + 1).clamp(max=PITY_CAP),
                                  rare_offset)
        rare_offset = torch.where(rarity == 2, torch.full_like(rare_offset, PITY_RESET),
                                  rare_offset)
        w = pool_mask[rarity] * (1.0 - taken)
        empty = w.sum(-1) < 0.5                     # pool exhausted by dedup
        w = torch.where(empty.unsqueeze(-1), pool_mask[rarity], w)
        pick = torch.multinomial(w.clamp(min=1e-9), 1).squeeze(-1)
        offers[:, k] = pick
        taken.scatter_(1, pick.unsqueeze(1), 1.0)
    return offers, rare_offset


def apply_pick(C, deck_counts, offers, choice):
    """choice (B,) in [0, N_OFFERS] where N_OFFERS = skip."""
    take = choice < N_OFFERS
    row = offers.gather(1, choice.clamp(max=N_OFFERS - 1).unsqueeze(1)).squeeze(1)
    add = F.one_hot(row, C.N).float() * take.float().unsqueeze(-1)
    return deck_counts + add
