"""Act 1 run simulator.

Floor plan (configurable): combats draw from the easy pool for the first 3
combats, then the hard pool (per selection_rules), with the last-2 repeat
exclusion; elite and boss floors draw from their pools. After every combat
victory: gold + a 3-card rarity offer (drop table + pity). Campfires: REST
(heal 30% of max HP) or upgrade one card (base -> upgraded row).

Deciders: drafter in {None, "random", MetaAgent}; policy is the combat agent
(None = uniform random legal play).
"""
import torch
import torch.nn.functional as F

from .engine import combat
from .rewards import roll_offers, apply_pick, init_pity, N_OFFERS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canon-paced Act 1 gauntlet: 8 fights, 3 campfires, 1 elite, boss.
DEFAULT_FLOORS = ["combat", "combat", "combat", "combat", "campfire",
                  "combat", "elite", "campfire", "combat", "combat",
                  "campfire", "boss"]
REST_HEAL_FRAC = 0.30


def _pick_encounters(C, pool_name, B, last1, last2):
    pool = C.POOLS[pool_name]
    ids = [p["encounter"] for p in pool]
    w = torch.tensor([float(p.get("weight", 1)) for p in pool], device=DEVICE)
    W = w.unsqueeze(0).repeat(B, 1)
    for j, eid in enumerate(ids):
        blocked = torch.tensor([(l1 == eid or l2 == eid)
                                for l1, l2 in zip(last1, last2)], device=DEVICE)
        W[:, j] = torch.where(blocked, torch.zeros_like(W[:, j]), W[:, j])
    W = torch.where(W.sum(-1, keepdim=True) < 1e-6,
                    w.unsqueeze(0).expand(B, -1), W)
    pick = torch.multinomial(W.clamp(min=1e-9), 1).squeeze(-1)
    return [ids[int(i)] for i in pick]


def _campfire(C, meta, deck, hp, alive, record=None):
    B = deck.shape[0]
    heal = REST_HEAL_FRAC * C.HP_MAX
    if meta is None or meta == "random":
        rest = (torch.rand(B, device=DEVICE) < 0.5)
        base = deck[:, :C.M]
        has = base.sum(-1) > 0.5
        pick = torch.multinomial(base.clamp(min=1e-9), 1).squeeze(-1)
        up = F.one_hot(pick, C.M).float() * ((~rest) & has).float().unsqueeze(-1)
    else:
        with torch.no_grad():
            logits = meta.campfire_logits(C, deck, hp)
            a = torch.distributions.Categorical(logits=logits).sample()
        rest = a == C.M
        up = F.one_hot(a.clamp(max=C.M - 1), C.M).float() * (~rest).float().unsqueeze(-1)
        up = up * (deck[:, :C.M].gather(1, a.clamp(max=C.M - 1).unsqueeze(1)) > 0.5).float()
    if record is not None:
        record["camp_hp"].append(hp.clone().cpu())
        record["camp_rest"].append(rest.float().cpu())
        record["upgrade"] += torch.cat([up.sum(0), rest.float().sum().unsqueeze(0)]).cpu()
    deck = deck + torch.cat([-up, up, torch.zeros(B, C.NS, device=DEVICE)], -1)
    hp = torch.minimum(hp + heal * rest.float() * alive,
                       torch.full_like(hp, C.HP_MAX))
    return deck, hp


def _draft(C, drafter, deck, hp, offers, alive, record=None):
    B = deck.shape[0]
    if drafter is None:
        choice = torch.full((B,), N_OFFERS, dtype=torch.long, device=DEVICE)
    elif drafter == "random":
        choice = torch.randint(0, N_OFFERS + 1, (B,), device=DEVICE)
    else:
        with torch.no_grad():
            logits = drafter.draft_logits(C, offers, deck, hp)
            choice = torch.distributions.Categorical(logits=logits).sample()
    choice = torch.where(alive > 0.5, choice,
                         torch.full_like(choice, N_OFFERS))
    if record is not None:
        take = choice < N_OFFERS
        row = offers.gather(1, choice.clamp(max=N_OFFERS - 1).unsqueeze(1)).squeeze(1)
        record["draft"][:-1] += torch.bincount(row[take], minlength=C.N).cpu().float()
        record["draft"][-1] += (~take).sum().cpu().float()
    return apply_pick(C, deck, offers, choice)


def simulate_runs(C, policy, drafter, B=512, floors=None, seed=0, record=False):
    torch.manual_seed(seed)
    floors = floors or DEFAULT_FLOORS
    deck = C.STARTER.unsqueeze(0).repeat(B, 1)
    hp = torch.full((B,), C.HP_MAX, device=DEVICE)
    alive = torch.ones(B, device=DEVICE)
    pity = init_pity(B)
    gold = torch.full((B,), 99.0, device=DEVICE)
    last1, last2 = [""] * B, [""] * B
    n_combats = 0
    rec = dict(draft=torch.zeros(C.N + 1), upgrade=torch.zeros(C.M + 1),
               camp_hp=[], camp_rest=[], floor_alive=[],
               snapshots=[]) if record else None
    for floor in floors:
        if floor == "campfire":
            deck, hp = _campfire(C, drafter, deck, hp, alive, rec)
            continue
        if floor == "combat":
            pool = "easy_hallway" if n_combats < int(
                C.SELECTION["normal_encounters"]["easy_pool_count"]) else "hard_hallway"
            enc = _pick_encounters(C, pool, B, last1, last2)
            last1, last2 = enc, last1
            source = "normal"
            n_combats += 1
        elif floor == "elite":
            enc = _pick_encounters(C, "elite", B, [""] * B, [""] * B)
            source = "elite"
        elif floor == "boss":
            enc = _pick_encounters(C, "boss", B, [""] * B, [""] * B)
            source = "boss"
        else:
            raise ValueError(floor)
        if record:
            rec["snapshots"].append(dict(deck=deck.clone(),
                                         hp=hp.clamp(min=1.0).clone(),
                                         enc=list(enc), source=floor,
                                         alive=alive.clone()))
        out = combat(C, deck, hp.clamp(min=1.0), enc, policy=policy)
        alive = alive * out["won"]
        hp = out["end_hp"] * alive
        gold = (gold - out["gold_lost"] + 15.0 * out["won"]).clamp(min=0)
        if record:
            rec["floor_alive"].append(alive.mean().item())
        if floor == "boss":
            break
        offers, pity = roll_offers(C, B, source, pity)
        deck = _draft(C, drafter, deck, hp.clamp(min=1.0), offers, alive, rec)
    result = dict(win=alive.mean().item(),
                  final_hp=(hp * alive).sum().item() / max(alive.sum().item(), 1),
                  gold=(gold * alive).sum().item() / max(alive.sum().item(), 1))
    if record:
        result.update(rec)
    return result
