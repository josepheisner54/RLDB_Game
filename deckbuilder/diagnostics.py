"""Diagnostics: the instrument panel.

These probes have caught more bugs than anything else in this project
(the mask exploit, V coverage holes, deck-check difficulty). Run them
after every training or content change:

    baseline_comparison(policy)          # is the win rate actually skill?
    decision_value_table(policy)         # do drafts/upgrades move outcomes?
    rest_probe(policy, meta)             # campfire behavior + draft taste
"""
import torch
import torch.nn.functional as F
from .state import S, DEVICE
from .engine.sampling import sample_enemy_conditions
from .engine.combat import combat
from .agents.policies import RandomPolicy
from .training.combat import eval_combat
from .runs import simulate_runs


def baseline_comparison(policy, B=4000):
    """A win rate means nothing without the random-play number beside it."""
    rand = RandomPolicy().to(DEVICE)
    t, r = eval_combat(policy, B=B), eval_combat(rand, B=B)
    print(f"trained {t['win']:.3f} (dHP {t['dhp']:+.1f})  vs  "
          f"random-play {r['win']:.3f} (dHP {r['dhp']:+.1f})")
    return dict(trained=t, random=r)


def deck_variant(B, drafts=0, upgrades=0):
    """Starter + k random drafts + k random upgrades, as a probability vector."""
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    for _ in range(drafts):
        counts += F.one_hot(torch.randint(0, S.M, (B,), device=DEVICE), S.N).float()
    for _ in range(upgrades):
        base = counts[:, :S.M]
        pick = torch.multinomial(base.clamp(min=1e-8), 1).squeeze(-1)
        has = (base.gather(1, pick.unsqueeze(1)).squeeze(1) >= 1).float()
        d = F.one_hot(pick, S.M).float() * has.unsqueeze(-1)
        counts = counts + torch.cat([-d, d], -1)
    return counts / counts.sum(-1, keepdim=True)


def decision_value_table(policy, B=3000, entering_hp=60.0,
                         scales=(1.0, 1.3, 1.6, 1.9, 2.2)):
    """Win rate by enemy scale x deck quality. If the columns don't separate,
    meta decisions have nothing to buy and the run is mistuned."""
    variants = [("starter", 0, 0), ("+2 drafts", 2, 0), ("+2dr+2up", 2, 2)]
    header = f"{'hp_scale':>9} | " + " | ".join(f"{n:>9}" for n, _, _ in variants)
    print(header)
    table = {}
    for hs in scales:
        row = []
        for name, dr, up in variants:
            torch.manual_seed(int(hs * 100) + dr * 10 + up)
            deck = deck_variant(B, dr, up)
            etype, hp0, ats = sample_enemy_conditions(
                B, hp_scale=hs, atk_scale=1.0 + (hs - 1) * 0.2)
            with torch.no_grad():
                out = combat(policy, deck, etype, hp0, ats,
                             torch.full((B,), entering_hp, device=DEVICE))
            row.append(out["won"].mean().item())
        table[hs] = row
        print(f"{hs:>9} | " + " | ".join(f"{v:>9.2f}" for v in row))
    return table


def rest_probe(policy, meta, B=2000, bins=((1, 30), (30, 45), (45, 70))):
    """Campfire behavior: P(rest) by entering HP, plus draft/upgrade taste.
    HP-conditional resting was a v3a success criterion; watch for bang-bang
    (all-0 or all-1) -- that means the rest value is outside the tradeoff
    band (see encounters.json: campfire_rest_heal)."""
    r = simulate_runs(policy, meta, B=B, record=True)
    print(f"run win {r['win']:.3f}   avg final HP {r['final_hp']:.1f}")
    if len(r["camp_hp"]) == 0:
        print("(no runs reached the campfire -- fights 1-2 are too hard)")
        return r
    hp = torch.cat(r["camp_hp"]); rest = torch.cat(r["camp_rest"])
    for lo, hi in bins:
        m = (hp >= lo) & (hp < hi)
        if m.sum() > 20:
            print(f"P(rest | HP {lo:>2}-{hi:<2}) = {rest[m].mean():.2f}   (n={int(m.sum())})")
    d = r["draft"][:S.N] / r["draft"].sum().clamp(min=1e-8)
    top_d = sorted(zip(S.NAMES, d.tolist()), key=lambda x: -x[1])[:5]
    print("top drafts:", {n: round(v, 2) for n, v in top_d})
    u = r["upgrade"] / r["upgrade"].sum().clamp(min=1e-8)
    top_u = sorted(zip(S.NAMES[:S.M] + ["REST"], u.tolist()), key=lambda x: -x[1])[:5]
    print("campfire:", {n: round(v, 2) for n, v in top_u})
    return r
