"""Training for the STS engine.

Combat agent: REINFORCE with a 3-head ValueNet baseline (dHP, P(death),
frac of enemy HP dealt). The frac term is dense credit where wins saturate
(bosses) -- weighted at SHAPE_FRAC HP-equivalents, deliberately small next
to the death penalty so it can't breed glass cannons.

Optimizer hygiene: AdamW(wd=1e-2) + linear warmup -> cosine decay, entropy
bonus (exploration collapse is RL's LR pathology), advantage normalization.

Stages: train_combat on broad sampled conditions -> finetune_on_runs on a
50/50 mix of harvested real-run snapshots and broad samples (never 100%
harvested; that is how coverage holes are born).
"""
import math
import torch
import torch.nn.functional as F

from .engine import combat
from .rewards import roll_offers, init_pity
from .agents import CombatPolicy, RandomPolicy, ValueNet, MetaAgent, \
    encounter_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_POOL = [("easy_hallway", 0.25), ("hard_hallway", 0.40),
              ("elite", 0.15), ("boss", 0.20)]
DEATH_PEN = 60.0
SHAPE_FRAC = 15.0        # HP-equivalent weight of the frac term
ENT_COEF = 0.01


def sample_conditions(C, B, pool_weights=None):
    pools, probs = zip(*(pool_weights or TRAIN_POOL))
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
    return dict(win=out["won"].mean().item(), dhp=out["delta_hp"].mean().item(),
                frac=out["frac"].mean().item())


def make_optim(params, lr, total_steps, warmup_frac=0.05, weight_decay=1e-2):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warm = max(int(total_steps * warmup_frac), 1)

    def sched(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, sched)


def _rl_step(C, policy, V, opt, sched, optv, deck, enc, hp, ef, eidx,
             ent_coef=ENT_COEF):
    out = combat(C, deck, hp, enc, policy=policy, collect_logp=True)
    died = 1 - out["won"]
    R = (out["delta_hp"] - DEATH_PEN * died + SHAPE_FRAC * out["frac"]) / 80.0
    encf = ef[torch.tensor([eidx[e] for e in enc], device=DEVICE)]
    with torch.no_grad():
        vd, vp, vf = V(hp, encf, deck)
        baseline = (vd - DEATH_PEN * vp + SHAPE_FRAC * vf) / 80.0
    adv = (R - baseline).detach()
    adv = (adv - adv.mean()) / adv.std().clamp(min=1e-6)
    loss = -(adv * out["logp"]).mean() - ent_coef * out["entropy"].mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    opt.step(); sched.step()
    vd, vp, vf = V(hp, encf, deck)
    vloss = F.mse_loss(vd / 80, out["delta_hp"] / 80) \
        + F.binary_cross_entropy(vp, died) \
        + F.mse_loss(vf, out["frac"])
    optv.zero_grad(); vloss.backward(); optv.step()
    return out


def train_combat(C, steps=600, B=128, eval_every=50, lr_pi=1e-3, lr_v=1e-3,
                 policy=None, V=None, verbose=True):
    policy = policy or CombatPolicy(C).to(DEVICE)
    V = V or ValueNet(C).to(DEVICE)
    ef, eidx = encounter_features(C)
    opt, sched = make_optim(policy.parameters(), lr_pi, steps)
    optv = torch.optim.Adam(V.parameters(), lr=lr_v)
    hist = []
    for s in range(steps):
        deck = sample_decks(C, B)
        enc = sample_conditions(C, B)
        hp = torch.rand(B, device=DEVICE) * 65 + 12
        out = _rl_step(C, policy, V, opt, sched, optv, deck, enc, hp, ef, eidx)
        if s % eval_every == 0 or s == steps - 1:
            ev = eval_combat(C, policy)
            hist.append((s, out["won"].mean().item(), ev["win"], ev["dhp"]))
            if verbose:
                print(f"[sts combat] step {s:4d}  batch win {out['won'].mean():.3f}  "
                      f"EVAL win {ev['win']:.3f}  dHP {ev['dhp']:+.1f}  "
                      f"frac {ev['frac']:.2f}")
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
        vd, vp, vf = V(hp, encf, deck)
        loss = F.mse_loss(vd / 80, out["delta_hp"] / 80) \
            + F.binary_cross_entropy(vp, 1 - out["won"]) \
            + F.mse_loss(vf, out["frac"])
        opt.zero_grad(); loss.backward(); opt.step()
    if verbose:
        print(f"[sts value] dHP MAE {(vd - out['delta_hp']).abs().mean():.2f} HP  "
              f"frac MAE {(vf - out['frac']).abs().mean():.3f}")
    return V


def finetune_on_runs(C, policy, V, meta, steps=300, B=128, harvest_runs=3,
                     harvest_B=512, mix=0.5, lr_pi=3e-4, eval_every=50,
                     verbose=True):
    """Stage 2: fine-tune the combat agent on the decks the meta agent
    ACTUALLY builds (harvested from full runs), mixed 50/50 with broad
    samples to protect coverage."""
    from .runsim import simulate_runs
    snaps = []
    for r in range(harvest_runs):
        rec = simulate_runs(C, policy, meta, B=harvest_B, seed=1000 + r,
                            record=True)
        snaps.extend(rec["snapshots"])
    if verbose:
        print(f"[finetune] harvested {len(snaps)} combat snapshots "
              f"({sum(int(s['alive'].sum()) for s in snaps)} live rows)")
    ef, eidx = encounter_features(C)
    opt, sched = make_optim(policy.parameters(), lr_pi, steps)
    optv = torch.optim.Adam(V.parameters(), lr=5e-4)
    hist = []
    for s in range(steps):
        if torch.rand(1).item() < mix and snaps:
            snap = snaps[int(torch.randint(0, len(snaps), (1,)))]
            live = torch.nonzero(snap["alive"] > 0.5).squeeze(-1)
            if len(live) == 0:
                continue
            rows = live[torch.randint(0, len(live), (B,), device=live.device)]
            deck = snap["deck"][rows]
            hp = snap["hp"][rows]
            enc = [snap["enc"][int(i)] for i in rows]
        else:
            deck = sample_decks(C, B)
            enc = sample_conditions(C, B)
            hp = torch.rand(B, device=DEVICE) * 65 + 12
        out = _rl_step(C, policy, V, opt, sched, optv, deck, enc, hp, ef, eidx)
        if s % eval_every == 0 or s == steps - 1:
            ev = eval_combat(C, policy)
            hist.append((s, ev["win"], ev["dhp"]))
            if verbose:
                print(f"[finetune] step {s:4d}  EVAL win {ev['win']:.3f}  "
                      f"dHP {ev['dhp']:+.1f}")
    return policy, V, hist


def _pool_mean_feats(C, ef, eidx, pool):
    ids = [p["encounter"] for p in C.POOLS[pool]]
    return ef[torch.tensor([eidx[e] for e in ids], device=DEVICE)].mean(0)


def train_meta(C, V, steps=400, B=512, eval_every=100, lr=1e-3, verbose=True):
    """Draft + campfire heads by backprop through frozen V. The synthetic
    future matches the canon-length act: hard fights, an elite, and the
    boss. Frac enters the objective so decisions keep gradient even where
    P(death) saturates."""
    from .rewards import N_OFFERS
    meta = MetaAgent(C).to(DEVICE)
    opt = torch.optim.Adam(meta.parameters(), lr=lr)
    ef, eidx = encounter_features(C)
    future = torch.stack([_pool_mean_feats(C, ef, eidx, p)
                          for p in ("hard_hallway", "hard_hallway", "elite",
                                    "hard_hallway", "boss")])
    pity = init_pity(B)
    hist = []

    def value_of(deck, hp, start=0):
        total = 0.0
        for f in range(start, future.shape[0]):
            vd, vp, vf = V(hp, future[f].unsqueeze(0).expand(deck.shape[0], -1),
                           deck)
            total = total + vd - DEATH_PEN * vp + SHAPE_FRAC * vf
        return total

    for s in range(steps):
        deck = sample_decks(C, B, max_drafts=4)
        hp = torch.rand(B, device=DEVICE) * 55 + 20
        offers, _ = roll_offers(C, B, "normal", pity)
        w = F.gumbel_softmax(meta.draft_logits(C, offers, deck, hp), tau=1.0)
        add = (w[:, :N_OFFERS].unsqueeze(-1)
               * F.one_hot(offers, C.N).float()).sum(1)
        vd_ = value_of(deck + add, hp)

        deck2 = sample_decks(C, B, max_drafts=4)
        hp2 = torch.rand(B, device=DEVICE) * 60 + 12
        w2 = F.gumbel_softmax(meta.campfire_logits(C, deck2, hp2), tau=1.0)
        up = w2[:, :C.M] * (deck2[:, :C.M] > 0.5).float()
        d2 = deck2 + torch.cat([-up, up,
                                torch.zeros(B, C.NS, device=DEVICE)], -1)
        h2 = torch.minimum(hp2 + w2[:, C.M] * 0.30 * C.HP_MAX,
                           torch.full_like(hp2, C.HP_MAX))
        vc_ = value_of(d2, h2, start=1)

        loss = -(vd_.mean() + vc_.mean()) / 200.0
        opt.zero_grad(); loss.backward(); opt.step()
        if s % eval_every == 0 or s == steps - 1:
            hist.append((s, -loss.item() * 200))
            if verbose:
                print(f"[sts meta] step {s:4d}  predicted value {-loss.item() * 200:+.1f}")
    return meta, hist
