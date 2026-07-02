"""Training for the STS engine.

Combat agent: REINFORCE with the ValueNet as a condition-aware baseline.
Conditions sample realistic mid-run states: starter + reward-rolled drafts
(rarity-correct), some upgrades, occasional Wound pollution, HP over the
FULL reachable range (coverage lesson: any input a decision can move must
be covered by V's training distribution).

Meta agent: backprop through the frozen V, over draft and campfire ops.
"""
import torch
import torch.nn.functional as F

from .engine import combat
from .rewards import roll_offers, init_pity
from .agents import CombatPolicy, RandomPolicy, ValueNet, MetaAgent, \
    encounter_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_POOL = [("easy_hallway", 0.30), ("hard_hallway", 0.40),
              ("elite", 0.15), ("boss", 0.15)]


def sample_conditions(C, B):
    pools, probs = zip(*TRAIN_POOL)
    which = torch.multinomial(torch.tensor(probs, device=DEVICE), B,
                              replacement=True)
    enc_ids = []
    for i in range(B):
        pool = C.POOLS[pools[which[i]]]
        w = torch.tensor([float(p.get("weight", 1)) for p in pool])
        enc_ids.append(pool[int(torch.multinomial(w, 1))]["encounter"])
    return enc_ids


def sample_decks(C, B, max_drafts=8, upgrade_p=0.4, wound_p=0.15):
    deck = C.STARTER.unsqueeze(0).repeat(B, 1)
    k = torch.randint(0, max_drafts + 1, (B,), device=DEVICE)
    pity = init_pity(B)
    for i in range(max_drafts):
        offers, pity = roll_offers(C, B, "normal", pity)
        pick = offers[:, 0]
        add = F.one_hot(pick, C.N).float() * (k > i).float().unsqueeze(-1)
        deck = deck + add
    for _ in range(2):
        do = (torch.rand(B, device=DEVICE) < upgrade_p).float()
        base = deck[:, :C.M]
        pick = torch.multinomial(base.clamp(min=1e-9), 1).squeeze(-1)
        has = (base.gather(1, pick.unsqueeze(1)).squeeze(1) >= 1).float() * do
        d = F.one_hot(pick, C.M).float() * has.unsqueeze(-1)
        deck = deck + torch.cat([-d, d, torch.zeros(B, C.NS, device=DEVICE)], -1)
    wounds = (torch.rand(B, device=DEVICE) < wound_p).float() \
        * torch.randint(1, 3, (B,), device=DEVICE).float()
    deck[:, C.card_row["wound"]] += wounds
    return deck


def eval_combat(C, policy, B=512, seed=123):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        deck = sample_decks(C, B)
        enc = sample_conditions(C, B)
        hp = torch.rand(B, device=DEVICE) * 50 + 25
        with torch.no_grad():
            out = combat(C, deck, hp, enc, policy=policy)
    return dict(win=out["won"].mean().item(), dhp=out["delta_hp"].mean().item())


def train_combat(C, steps=600, B=128, eval_every=50, lr_pi=1e-3, lr_v=1e-3,
                 policy=None, V=None, verbose=True):
    policy = policy or CombatPolicy(C).to(DEVICE)
    V = V or ValueNet(C).to(DEVICE)
    ef, eidx = encounter_features(C)
    opt = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    optv = torch.optim.Adam(V.parameters(), lr=lr_v)
    hist = []
    for s in range(steps):
        deck = sample_decks(C, B)
        enc = sample_conditions(C, B)
        hp = torch.rand(B, device=DEVICE) * 65 + 12
        out = combat(C, deck, hp, enc, policy=policy, collect_logp=True)
        died = 1 - out["won"]
        R = (out["delta_hp"] - 60.0 * died) / 80.0
        encf = ef[torch.tensor([eidx[e] for e in enc], device=DEVICE)]
        with torch.no_grad():
            vd, vp = V(hp, encf, deck)
            baseline = (vd - 60.0 * vp) / 80.0
        loss = -((R - baseline).detach() * out["logp"]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()
        vd, vp = V(hp, encf, deck)
        vloss = F.mse_loss(vd / 80, out["delta_hp"] / 80) \
            + F.binary_cross_entropy(vp, died)
        optv.zero_grad(); vloss.backward(); optv.step()
        if s % eval_every == 0 or s == steps - 1:
            ev = eval_combat(C, policy)
            hist.append((s, out["won"].mean().item(), ev["win"], ev["dhp"]))
            if verbose:
                print(f"[sts combat] step {s:4d}  batch win {out['won'].mean():.3f}  "
                      f"EVAL win {ev['win']:.3f}  dHP {ev['dhp']:+.1f}")
    return policy, V, hist


def refine_value(C, policy, V, steps=200, B=256, lr=5e-4, verbose=True):
    ef, eidx = encounter_features(C)
    opt = torch.optim.Adam(V.parameters(), lr=lr)
    for s in range(steps):
        deck = sample_decks(C, B)
        enc = sample_conditions(C, B)
        hp = torch.rand(B, device=DEVICE) * 75 + 5      # full reachable range
        with torch.no_grad():
            out = combat(C, deck, hp, enc, policy=policy)
        encf = ef[torch.tensor([eidx[e] for e in enc], device=DEVICE)]
        vd, vp = V(hp, encf, deck)
        loss = F.mse_loss(vd / 80, out["delta_hp"] / 80) \
            + F.binary_cross_entropy(vp, 1 - out["won"])
        opt.zero_grad(); loss.backward(); opt.step()
    if verbose:
        print(f"[sts value] dHP MAE {(vd - out['delta_hp']).abs().mean():.2f} HP")
    return V


def _pool_mean_feats(C, ef, eidx, pool):
    ids = [p["encounter"] for p in C.POOLS[pool]]
    return ef[torch.tensor([eidx[e] for e in ids], device=DEVICE)].mean(0)


def train_meta(C, V, steps=400, B=512, eval_every=100, lr=1e-3, verbose=True):
    """Draft + campfire heads by backprop through frozen V, valued against
    the remaining-run gauntlet (hard combat + elite + boss pool means)."""
    from .rewards import N_OFFERS
    meta = MetaAgent(C).to(DEVICE)
    opt = torch.optim.Adam(meta.parameters(), lr=lr)
    ef, eidx = encounter_features(C)
    future = torch.stack([_pool_mean_feats(C, ef, eidx, p)
                          for p in ("hard_hallway", "hard_hallway", "elite", "boss")])
    pity = init_pity(B)
    hist = []

    def value_of(deck, hp, start=0):
        total = 0.0
        for f in range(start, future.shape[0]):
            vd, vp = V(hp, future[f].unsqueeze(0).expand(deck.shape[0], -1), deck)
            total = total + vd - 60.0 * vp
        return total

    for s in range(steps):
        deck = sample_decks(C, B, max_drafts=3)
        hp = torch.rand(B, device=DEVICE) * 55 + 20
        offers, _ = roll_offers(C, B, "normal", pity)
        w = F.gumbel_softmax(meta.draft_logits(C, offers, deck, hp), tau=1.0)
        add = (w[:, :N_OFFERS].unsqueeze(-1)
               * F.one_hot(offers, C.N).float()).sum(1)
        vd_ = value_of(deck + add, hp)

        deck2 = sample_decks(C, B, max_drafts=3)
        hp2 = torch.rand(B, device=DEVICE) * 60 + 12
        w2 = F.gumbel_softmax(meta.campfire_logits(C, deck2, hp2), tau=1.0)
        up = w2[:, :C.M] * (deck2[:, :C.M] > 0.5).float()
        d2 = deck2 + torch.cat([-up, up,
                                torch.zeros(B, C.NS, device=DEVICE)], -1)
        h2 = torch.minimum(hp2 + w2[:, C.M] * 0.30 * C.HP_MAX,
                           torch.full_like(hp2, C.HP_MAX))
        vc_ = value_of(d2, h2, start=1)

        loss = -(vd_.mean() + vc_.mean()) / 160.0
        opt.zero_grad(); loss.backward(); opt.step()
        if s % eval_every == 0 or s == steps - 1:
            hist.append((s, -loss.item() * 160))
            if verbose:
                print(f"[sts meta] step {s:4d}  predicted value {-loss.item() * 160:+.1f}")
    return meta, hist
