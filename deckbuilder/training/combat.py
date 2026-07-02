"""Combat agent training: REINFORCE with V as a condition-aware baseline
(actor-critic). Under randomized conditions a batch-mean baseline is weak --
winning an EASY fight looks like good play."""
import torch
import torch.nn.functional as F
from ..state import DEVICE
from ..engine.sampling import sample_decks, sample_enemy_conditions, enemy_summary, jitter_feats
from ..engine.combat import combat
from ..agents.policies import make_policy
from ..agents.value import ValueNet


def eval_combat(policy, B=4000, seed=123):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        deck = sample_decks(B)
        etype, hp0, ats = sample_enemy_conditions(B)
        hp = torch.rand(B, device=DEVICE) * 45 + 25
        with torch.no_grad():
            out = combat(policy, deck, etype, hp0, ats, hp)
    return dict(win=out["won"].mean().item(), dhp=out["delta_hp"].mean().item())


def train_combat(steps=900, B=512, jitter=1.0, eval_every=100,
                 policy=None, V=None, lr_pi=2e-3, lr_v=1e-3,
                 deck_sampler=None, condition_sampler=None, verbose=True):
    policy = policy or make_policy("mlp").to(DEVICE)
    V = V or ValueNet().to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    optv = torch.optim.Adam(V.parameters(), lr=lr_v)
    decks = deck_sampler or sample_decks
    conds = condition_sampler or sample_enemy_conditions
    hist = []
    for s in range(steps):
        deck = decks(B)
        etype, hp0, ats = conds(B)
        hp = torch.rand(B, device=DEVICE) * 55 + 15        # cover low-HP entries
        out = combat(policy, deck, etype, hp0, ats, hp,
                     feats=jitter_feats(B, jitter), collect_logp=True)
        died = 1 - out["won"]
        R = (out["delta_hp"] - 60.0 * died) / 70.0
        esum = enemy_summary(etype, hp0, ats)
        with torch.no_grad():
            vd, vp = V(hp, esum, deck)
            baseline = (vd - 60.0 * vp) / 70.0
        loss = -((R - baseline).detach() * out["logp"]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        vd, vp = V(hp, esum, deck)
        vloss = F.mse_loss(vd / 70, out["delta_hp"] / 70) + F.binary_cross_entropy(vp, died)
        optv.zero_grad(); vloss.backward(); optv.step()
        if s % eval_every == 0 or s == steps - 1:
            ev = eval_combat(policy)
            hist.append((s, out["won"].mean().item(), ev["win"], ev["dhp"]))
            if verbose:
                print(f"[combat] step {s:5d}  batch win {out['won'].mean():.3f}  "
                      f"EVAL win {ev['win']:.3f}  dHP {ev['dhp']:+.1f}")
    return policy, V, hist


def refine_value(policy, V, steps=300, B=1024, lr=5e-4, verbose=True):
    """V must see the FULL reachable HP range (down to 5): campfires visit
    low-HP states, and a V that never saw them extrapolates blindly.
    (Coverage holes: three appearances and counting.)"""
    opt = torch.optim.Adam(V.parameters(), lr=lr)
    for s in range(steps):
        deck = sample_decks(B)
        etype, hp0, ats = sample_enemy_conditions(B)
        hp = torch.rand(B, device=DEVICE) * 65 + 5
        with torch.no_grad():
            out = combat(policy, deck, etype, hp0, ats, hp)
        vd, vp = V(hp, enemy_summary(etype, hp0, ats), deck)
        loss = F.mse_loss(vd / 70, out["delta_hp"] / 70) + F.binary_cross_entropy(vp, 1 - out["won"])
        opt.zero_grad(); loss.backward(); opt.step()
    if verbose:
        print(f"[value] refined: dHP MAE {(vd - out['delta_hp']).abs().mean().item():.2f} HP")
    return V
