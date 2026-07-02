"""Game state registry: content JSONs compiled to tensors, reloadable in place.

Design note: a single mutable registry `S` is a deliberate pragmatic choice --
every engine function reads S at call time, so `load(content_dir)` swaps the
entire game (new cards, enemies, encounters) without re-importing anything.
If/when multi-game or parallel-content workflows appear, the refactor path is
to thread a GameContent object through the call graph.
"""
import json
from importlib import resources
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CARD_FIELDS = ["cost", "damage", "hits", "block", "vulnerable", "weak", "strength"]
COST, DMG, HITS, BLK, VULN, WEAK, STR = range(7)
INTENT_FIELDS = ["attack", "hits", "block", "strength", "heal"]
I_ATK, I_HITS, I_BLK, I_STR, I_HEAL = range(5)

CFG = dict(hp_max=70.0, turns=8, energy=3.0, hand_size=5, plays_per_turn=4)


class _State:
    pass


S = _State()


def load(content_dir=None):
    """(Re)compile content JSONs into tensors on the registry, in place."""
    if content_dir is None:
        base = resources.files("deckbuilder.content")
        read = lambda name: json.loads((base / name).read_text())
    else:
        read = lambda name: json.load(open(f"{content_dir}/{name}"))

    cards = read("cards.json")["cards"]
    M = len(cards)
    names, rows = [], []
    for c in cards:
        names.append(c["name"])
        rows.append([float(c[f]) for f in CARD_FIELDS])
    for c in cards:
        names.append(c["name"] + "+")
        merged = {**c, **c.get("upgrade", {})}
        rows.append([float(merged[f]) for f in CARD_FIELDS])
    S.M, S.N = M, 2 * M
    S.NAMES = names
    S.FEATS = torch.tensor(rows, dtype=torch.float32, device=DEVICE)

    enemies = read("enemies.json")["enemies"]
    E, K = len(enemies), max(len(e["intents"]) for e in enemies)
    S.E, S.K = E, K
    S.ENAMES = [e["name"] for e in enemies]
    S.E_HP = torch.tensor([e["hp"] for e in enemies], dtype=torch.float32, device=DEVICE)
    intents = torch.zeros(E, K, len(INTENT_FIELDS), device=DEVICE)
    weights = torch.zeros(E, K, device=DEVICE)
    n_int = torch.zeros(E, dtype=torch.long, device=DEVICE)
    is_cycle = torch.zeros(E, device=DEVICE)
    max_rep = torch.full((E,), 99, dtype=torch.long, device=DEVICE)
    for i, e in enumerate(enemies):
        n_int[i] = len(e["intents"])
        for j, it in enumerate(e["intents"]):
            intents[i, j] = torch.tensor([float(it[f]) for f in INTENT_FIELDS], device=DEVICE)
        p = e["pattern"]
        if p["type"] == "cycle":
            is_cycle[i] = 1.0
            weights[i, :n_int[i]] = 1.0 / n_int[i].float()
        else:
            weights[i, :n_int[i]] = torch.tensor(p["weights"], device=DEVICE)
            max_rep[i] = p.get("max_repeat", 99)
    S.INTENTS, S.WEIGHTS = intents, weights
    S.N_INT, S.IS_CYCLE, S.MAX_REP = n_int, is_cycle, max_rep
    ew = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
    S.E_SUMMARY = torch.stack([
        (ew * intents[..., I_ATK] * intents[..., I_HITS].clamp(min=1)).sum(-1),
        (ew * intents[..., I_BLK]).sum(-1),
        (ew * intents[..., I_STR]).sum(-1)], -1)

    enc = read("encounters.json")
    S.FIGHTS = enc["fights"]
    S.RUN = enc["run"]

    starter = torch.zeros(S.N, device=DEVICE)
    starter[S.NAMES.index("Strike")] = 5
    starter[S.NAMES.index("Defend")] = 4
    starter[S.NAMES.index("Bash")] = 1
    S.STARTER = starter
    return S


load()
