"""Compile the STS JSON bundle into batched-executable tensors.

Contract: unknown ops, powers, dynamics, or AI fields raise immediately at
load time. Coverage is proven, not assumed. Ascension is resolved here
(by_ascension maps collapse to scalars), so changing ascension = reload.

Documented approximations (each touches <=2 cards/enemies):
- piles are count-based (unordered); "top of draw pile" ops act on a random
  draw-pile card; Headbutt puts the moved card in the draw pile (not on top)
- "choose a card" effects (Armaments, True Grit, Dual Wield, Warcry) act on
  a random eligible card instead of an agent-chosen one
- Looter's lunge/smoke branch is compiled to its two sequences behind a coin
  flip at the branch point
"""
import json
from importlib import resources
import torch

from .vocab import (OP, PW, P, AMT_X, AMT_PLAYER_BLOCK, AMT_RAMPAGE, AMT_PERFECTED, AMT_SPAWN,
                    AMT_SOURCE_HP, AMT_CAPTURED, AMT_DIVIDER,
                    CARD_DYNAMIC_BURN, TGT_CHOSEN, TGT_PLAYER,
                    TGT_ALL_ENEMIES, TGT_RANDOM_EACH_HIT, TGT_SELF, TGT_ALLY,
                    CND_NONE, CND_TGT_VULN, CND_TGT_INTENT_ATK, CND_ASC17,
                    CND_FATAL, DEST_DRAW, DEST_DISCARD, DEST_HAND,
                    CT_ATTACK, CT_SKILL, CT_POWER, CT_STATUS, RARITY,
                    AI_WEIGHTED, AI_SEQUENCE, AI_ALTERNATING,
                    AI_SHIELD_GREMLIN, AI_LOOTER, AI_LAGAVULIN, AI_GUARDIAN,
                    AI_CYCLE_INTERRUPT, INTENT_ATTACK, INTENT_DEFEND,
                    INTENT_BUFF, INTENT_DEBUFF, INTENT_OTHER,
                    MAX_FX, NPARAM, E_MAX)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _asc(v, ascension):
    """Resolve a by-ascension map {level: value} to a scalar."""
    if isinstance(v, dict) and v and all(str(k).isdigit() for k in v):
        best = max((int(k) for k in v if int(k) <= ascension), default=None)
        if best is None:
            best = min(int(k) for k in v)
        return v[str(best)] if str(best) in v else v[best]
    return v


class StsContent:
    pass


def _amount(v, ascension, card_id=""):
    v = _asc(v, ascension)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "dynamic" in v:
            d = v["dynamic"]
            return {"energy_spent_x": AMT_X, "player.block": AMT_PLAYER_BLOCK,
                    "this_card.combat_damage": AMT_RAMPAGE,
                    "source.current_hp": AMT_SOURCE_HP}[d]
        if "captured" in v:
            return AMT_CAPTURED
        if "snapshot_at_intent_selection" in v:
            return AMT_DIVIDER          # hexaghost divider formula
        if "spawn_roll" in v:
            return AMT_SPAWN             # per-slot roll (see spawn_rolls)
        if "plus_per_card_name_contains" in v:
            return AMT_PERFECTED         # param7 carries (base, per-card bonus)
    raise ValueError(f"unknown amount {v!r} in {card_id}")


def _target(v, source_is_enemy):
    if isinstance(v, dict):
        assert v.get("dynamic") == "chosen_living_ally"
        return TGT_ALLY
    return {"enemy": TGT_CHOSEN, "player": TGT_PLAYER,
            "all_enemies": TGT_ALL_ENEMIES,
            "random_enemy_each_hit": TGT_RANDOM_EACH_HIT,
            "self": TGT_SELF, "self_instance": TGT_SELF}[v]


def _compile_effects(effects, C, ascension, source_is_enemy, cid,
                     default_target, exhaust_kw):
    """One JSON effect list -> list of MAX_FX param rows."""
    rows, cond = [], CND_NONE
    for fx in effects:
        op = fx["op"]
        r = [0.0] * NPARAM
        r[6] = float(cond)
        amt = fx.get("amount", 0)
        hits = _asc(fx.get("hits", 1), ascension)
        if isinstance(hits, dict):
            hits = AMT_X                 # Whirlwind: X hits
        tgt = _target(fx["target"], source_is_enemy) if "target" in fx else default_target

        if op == "conditional":
            cnd = fx["if_"]
            if cnd == {"target_has_power": "vulnerable"}:
                code = CND_TGT_VULN
            elif cnd == {"target_intent_has_attack": True}:
                code = CND_TGT_INTENT_ATK
            elif "ascension_gte" in cnd:
                code = CND_ASC17 if ascension >= cnd["ascension_gte"] else -1
            else:
                raise ValueError(f"unknown condition {cnd} in {cid}")
            if code == -1:
                continue                 # condition statically false
            for sub in fx["then"]:
                rows.extend(_compile_effects([sub], C, ascension,
                                             source_is_enemy, cid,
                                             default_target, exhaust_kw))
                rows[-1][6] = float(code) if code != CND_ASC17 else CND_NONE
            continue
        if op == "fatal":
            cond = CND_FATAL             # gates the REST of this program
            continue
        if op == "capture":
            r[0] = OP["capture"]
            r[5] = 1.0 if fx.get("field") == "current_hp" else 0.0
            rows.append(r); continue
        if op in ("damage", "damage_self"):
            r[0] = OP["damage_self" if op == "damage_self" else "damage"]
            r[1] = _amount(amt, ascension, cid)
            if r[1] == AMT_PERFECTED:
                r[7] = float(amt["base"])
                r[5] = float(amt["plus_per_card_name_contains"]["amount"])
            if r[1] == AMT_SPAWN and isinstance(amt, dict):
                r[7] = float(_asc({"0": 0, **amt.get("ascension_add", {})}, ascension))
            r[2] = float(hits)
            r[3] = float(_asc(fx.get("strength_multiplier", 1), ascension))
            r[4] = float(TGT_SELF if op == "damage_self" else tgt)
            rows.append(r); continue
        if op == "gain_block":
            r[0] = OP["gain_block"]; r[1] = _amount(amt, ascension, cid)
            r[4] = float(tgt if "target" in fx else (TGT_SELF if source_is_enemy else TGT_PLAYER))
            rows.append(r); continue
        if op == "apply_power":
            pname, a = fx["power"], _amount(amt, ascension, cid)
            if pname == "strength_down":
                pname, a = "strength", -a
            if pname not in PW:
                raise ValueError(f"unknown power {pname} in {cid}")
            r[0] = OP["apply_power"]; r[1] = a; r[3] = float(PW[pname])
            r[4] = float(tgt)
            rows.append(r); continue
        if op in ("draw", "gain_energy", "lose_hp", "heal", "gain_max_hp",
                  "steal_gold", "multiply_block", "multiply_power",
                  "set_flag", "rampage_grow"):
            r[0] = OP[op]; r[1] = _amount(amt, ascension, cid) if amt else float(_asc(fx.get("factor", fx.get("amount", 2)), ascension))
            if op == "multiply_power":
                r[3] = float(PW[fx.get("power", "strength")])
            rows.append(r); continue
        if op == "modify_card_for_combat":       # Rampage
            r[0] = OP["rampage_grow"]; r[1] = _amount(fx.get("increase_damage_by", fx.get("amount", 5)), ascension, cid)
            rows.append(r); continue
        if op == "create_card":
            card = fx["card"]
            if isinstance(card, dict):
                row_id = CARD_DYNAMIC_BURN       # hexaghost burn/burn+
            else:
                row_id = C.card_row[card]
            r[0] = OP["create_card"]; r[1] = _amount(fx.get("amount", 1), ascension, cid)
            r[3] = float(row_id)
            r[5] = float({"draw_pile": DEST_DRAW, "discard_pile": DEST_DISCARD,
                          "hand": DEST_HAND}[fx.get("destination", "discard_pile")])
            rows.append(r); continue
        if op in ("create_copy", "copy_card"):
            r[0] = OP["copy_in_hand"]; r[1] = float(_asc(fx.get("copies", fx.get("amount", 1)), ascension))
            rows.append(r); continue
        if op == "generate_random_card":
            r[0] = OP["generate_random_card"]; r[1] = float(_asc(fx.get("amount", 1), ascension))
            rows.append(r); continue
        if op == "exhaust_self":
            r[0] = OP["exhaust_self"]; rows.append(r); continue
        if op == "exhaust_card":
            r[0] = OP["exhaust_random"]; r[1] = float(_asc(fx.get("amount", 1), ascension))
            r[5] = float(fx.get("capture_count", 0) or ("capture" in json.dumps(fx)))
            rows.append(r); continue
        if op == "exhaust_cards":
            sel = fx.get("selection", fx.get("filter", "hand"))
            r[0] = OP["exhaust_hand"]
            r[5] = 1.0 if "non_attack" in json.dumps(sel) else 0.0
            r[1] = float(_asc(fx.get("amount", -1), ascension) or -1)
            rows.append(r); continue
        if op == "gain_block_per_result":
            r[0] = OP["block_per_captured"]; r[1] = _amount(fx.get("block_per", fx.get("amount", 5)), ascension, cid)
            rows.append(r); continue
        if op == "move_card":
            src = fx.get("from", fx.get("source_zone", "discard_pile"))
            r[0] = OP["move_exhaust_to_hand" if "exhaust" in str(src) else "move_discard_to_draw"]
            r[1] = float(_asc(fx.get("amount", 1), ascension))
            rows.append(r); continue
        if op == "play_top_card":
            r[0] = OP["play_top_card"]; r[1] = float(_asc(fx.get("amount", 1), ascension))
            rows.append(r); continue
        if op in ("upgrade_cards", "upgrade_all_cards"):
            r[0] = OP["upgrade_in_hand"]
            r[1] = -1.0 if op == "upgrade_all_cards" else float(_asc(fx.get("amount", 1), ascension))
            rows.append(r); continue
        if op == "change_state":
            r[0] = OP["change_state"]; rows.append(r); continue
        if op == "spawn":
            r[0] = OP["spawn"]; r[1] = float(fx.get("count", 1))
            r[3] = -1.0                              # enemy id patched by caller
            ho = fx.get("hp_override", {})
            r[5] = 1.0 if ho else 0.0
            r[7] = 0.0
            rows.append((r, fx["enemy"])); continue  # tuple: needs enemy id
        if op == "die_without_rewards":
            r[0] = OP["die_no_rewards"]; rows.append(r); continue
        if op == "escape_with_stolen_gold":
            r[0] = OP["escape"]; rows.append(r); continue
        if op == "remove_power":
            r[0] = OP["remove_power"]; r[3] = float(PW[fx["power"]])
            rows.append(r); continue
        raise ValueError(f"unhandled op {op!r} in {cid}")
    if exhaust_kw and not any((rw[0] if not isinstance(rw, tuple) else rw[0][0]) == OP["exhaust_self"]
                              for rw in rows):
        rows.append([float(OP["exhaust_self"])] + [0.0] * (NPARAM - 1))
    assert len(rows) <= MAX_FX, f"{cid}: {len(rows)} effects > MAX_FX"
    return rows


def load(ascension=0, content_dir=None):
    C = StsContent()
    C.ascension = ascension
    if content_dir is None:
        base = resources.files("deckbuilder.sts.content")
        read = lambda n: json.loads((base / n).read_text())
    else:
        read = lambda n: json.load(open(f"{content_dir}/{n}"))

    cards = read("cards_ironclad.json")["cards"]
    statuses = read("status_cards.json")["cards"]
    ch = read("character_ironclad.json")
    M = len(cards)
    C.M, C.NS = M, len(statuses)
    C.N = 2 * M + C.NS
    C.card_row = {}
    C.NAMES = []
    for i, c in enumerate(cards):
        C.card_row[c["id"]] = i
        C.NAMES.append(c["name"])
    for i, c in enumerate(cards):
        C.NAMES.append(c["name"] + "+")
    for j, s in enumerate(statuses):
        C.card_row[s["id"]] = 2 * M + j
        C.NAMES.append(s["name"])

    prog = torch.zeros(C.N, MAX_FX, NPARAM)
    cost = torch.zeros(C.N); ctype = torch.zeros(C.N, dtype=torch.long)
    targeted = torch.zeros(C.N); exha = torch.zeros(C.N)
    ether = torch.zeros(C.N); innate = torch.zeros(C.N)
    unplay = torch.zeros(C.N); rar = torch.zeros(C.N, dtype=torch.long)
    clash = torch.zeros(C.N)
    eot_prog = torch.zeros(C.N, MAX_FX, NPARAM)      # burn end-of-turn
    TYPES = {"attack": CT_ATTACK, "skill": CT_SKILL, "power": CT_POWER,
             "status": CT_STATUS}

    def fill(row, cdef, variant):
        import copy
        v = copy.deepcopy(cdef[variant])
        n_up = 0 if variant == "base" else 1
        def resolve_formula(x):
            if isinstance(x, dict):
                if "formula" in x and x.get("n") == "upgrade_count":
                    return float(eval(x["formula"].replace("n", str(n_up))))
                return {k: resolve_formula(vv) for k, vv in x.items()}
            if isinstance(x, list):
                return [resolve_formula(vv) for vv in x]
            return x
        v = resolve_formula(v)
        kw = set(v.get("keywords", cdef.get("keywords", [])))
        cst = _asc(cdef["cost"][ "base" if variant == "base" else "upgraded"], ascension)
        cost[row] = -1.0 if cst == "X" else float(cst)
        ctype[row] = TYPES[cdef["type"]]
        targeted[row] = 1.0 if cdef.get("target") == "enemy" else 0.0
        exha[row] = 1.0 if "exhaust" in kw else 0.0
        ether[row] = 1.0 if "ethereal" in kw else 0.0
        innate[row] = 1.0 if "innate" in kw else 0.0
        rar[row] = RARITY[cdef["rarity"]]
        if cdef.get("playable_if") or "clash" in cdef["id"]:
            clash[row] = 1.0
        rows = _compile_effects(v["effects"], C, ascension, False, cdef["id"],
                                TGT_CHOSEN if targeted[row] else
                                (TGT_ALL_ENEMIES if cdef.get("target") == "all_enemies"
                                 else (TGT_RANDOM_EACH_HIT if cdef.get("target") == "random_enemy_each_hit"
                                       else TGT_PLAYER)),
                                "exhaust" in kw)
        for k, rw in enumerate(rows):
            assert not isinstance(rw, tuple), "cards do not spawn"
            prog[row, k] = torch.tensor(rw)

    for i, c in enumerate(cards):
        fill(i, c, "base")
        fill(M + i, c, "upgraded")
    for j, s in enumerate(statuses):
        row = 2 * M + j
        kw = set(s.get("keywords", []))
        cost[row] = -2.0 if "unplayable" in kw else float(s.get("cost") or 0)
        ctype[row] = CT_STATUS; rar[row] = RARITY["status"]
        ether[row] = 1.0 if "ethereal" in kw else 0.0
        unplay[row] = 1.0 if "unplayable" in kw else 0.0
        exha[row] = 1.0 if "exhaust" in kw else 0.0
        for k, rw in enumerate(_compile_effects(s.get("effects", []), C,
                                                ascension, False, s["id"],
                                                TGT_PLAYER, "exhaust" in kw)):
            prog[row, k] = torch.tensor(rw)
        for k, rw in enumerate(_compile_effects(s.get("end_of_turn", []), C,
                                                ascension, False, s["id"],
                                                TGT_PLAYER, False)):
            eot_prog[row, k] = torch.tensor(rw)

    C.PROG = prog.to(DEVICE); C.EOT_PROG = eot_prog.to(DEVICE)
    C.COST = cost.to(DEVICE); C.CTYPE = ctype.to(DEVICE)
    C.TARGETED = targeted.to(DEVICE); C.EXHAUST = exha.to(DEVICE)
    C.ETHEREAL = ether.to(DEVICE); C.INNATE = innate.to(DEVICE)
    C.UNPLAYABLE = unplay.to(DEVICE); C.RARITY_T = rar.to(DEVICE)
    C.CLASH = clash.to(DEVICE)
    C.BURN_ROW = C.card_row["burn"]; C.BURN_PLUS_ROW = C.card_row["burn_plus"]
    C.STRIKE_MASK = torch.tensor([1.0 if "Strike" in n else 0.0
                                  for n in C.NAMES]).to(DEVICE)
    C.ATTACK_ROWS = (C.CTYPE == CT_ATTACK)

    # agent-facing card features, derived from programs
    dmg_est = torch.zeros(C.N); hits_est = torch.ones(C.N)
    blk_est = torch.zeros(C.N); draw_est = torch.zeros(C.N)
    pw_est = torch.zeros(C.N); nrg_est = torch.zeros(C.N)
    for r in range(C.N):
        for k in range(MAX_FX):
            o = int(prog[r, k, 0]); a = float(prog[r, k, 1])
            if o == OP["damage"] and a > 0:
                dmg_est[r] += a; hits_est[r] = max(hits_est[r].item(), prog[r, k, 2].item())
            if o == OP["gain_block"] and a > 0: blk_est[r] += a
            if o == OP["draw"]: draw_est[r] += a
            if o == OP["apply_power"]: pw_est[r] += abs(a)
            if o == OP["gain_energy"]: nrg_est[r] += a
    tf = torch.nn.functional.one_hot(ctype, 4).float()
    C.CARD_FEATS = torch.cat([
        (cost.clamp(min=0) / 3).unsqueeze(1), (cost == -1).float().unsqueeze(1),
        (dmg_est / 15).unsqueeze(1), (hits_est / 4).unsqueeze(1),
        (blk_est / 12).unsqueeze(1), (draw_est / 3).unsqueeze(1),
        (pw_est / 4).unsqueeze(1), (nrg_est / 2).unsqueeze(1),
        tf, exha.unsqueeze(1), ether.unsqueeze(1), targeted.unsqueeze(1),
        (rar.float() / 4).unsqueeze(1), unplay.unsqueeze(1),
        torch.cat([torch.zeros(M), torch.ones(M), torch.zeros(C.NS)]).unsqueeze(1),
    ], dim=1).to(DEVICE)
    C.FEAT_DIM = C.CARD_FEATS.shape[1]

    # rarity pools for rewards (base rows only, no basics/statuses)
    C.POOL = {r: [i for i, c in enumerate(cards) if c["rarity"] == r]
              for r in ("common", "uncommon", "rare")}

    C.HP_MAX = float(ch["max_hp"]); C.ENERGY = float(ch["base_energy_per_turn"])
    C.DRAW = int(ch["base_draw_per_turn"]); C.HAND_MAX = int(ch["hand_size_limit"])
    starter = torch.zeros(C.N)
    for e in ch["starting_deck"]:
        starter[C.card_row[e["card"]]] = e["count"]
    C.STARTER = starter.to(DEVICE)
    C.BURNING_BLOOD_HEAL = 6.0

    _load_enemies(C, read("enemies_act1.json")["enemies"], ascension)
    _load_encounters(C, read("encounters_act1.json"))
    return C


def _load_enemies(C, enemies, ascension):
    C.E_TYPES = len(enemies)
    C.ENAMES = [e["id"] for e in enemies]
    C.eid = {e["id"]: i for i, e in enumerate(enemies)}
    C.E_HP = torch.stack([
        torch.tensor(_asc(e["hp_by_ascension"], ascension), dtype=torch.float32)
        for e in enemies]).to(DEVICE)

    move_rows, move_meta = [], []
    C.MOVE_BASE = torch.zeros(C.E_TYPES, dtype=torch.long)
    C.N_MOVES = torch.zeros(C.E_TYPES, dtype=torch.long)
    C.move_id = {}
    spawn_patch = []
    for ei, e in enumerate(enemies):
        C.MOVE_BASE[ei] = len(move_rows)
        for mi, (mid, m) in enumerate(e["moves"].items()):
            C.move_id[(e["id"], mid)] = mi
            pr = torch.zeros(MAX_FX, NPARAM)
            rows = _compile_effects(m.get("effects", []), C, ascension, True,
                                    f"{e['id']}.{mid}", TGT_PLAYER, False)
            for k, rw in enumerate(rows):
                if isinstance(rw, tuple):
                    rw, spawn_name = rw
                    spawn_patch.append((len(move_rows), k, spawn_name))
                pr[k] = torch.tensor(rw)
            move_rows.append(pr)
            intent = m.get("intent", "unknown")
            icode = (INTENT_ATTACK if "attack" in intent else
                     INTENT_DEFEND if "defend" in intent else
                     INTENT_BUFF if "buff" in intent else
                     INTENT_DEBUFF if "debuff" in intent else INTENT_OTHER)
            move_meta.append(icode)
        C.N_MOVES[ei] = len(e["moves"])
    C.MOVES = torch.stack(move_rows).to(DEVICE)
    for row, k, name in spawn_patch:
        C.MOVES[row, k, 3] = float(C.eid[name])
    C.MOVE_INTENT = torch.tensor(move_meta, dtype=torch.long).to(DEVICE)
    C.MOVE_BASE = C.MOVE_BASE.to(DEVICE); C.N_MOVES = C.N_MOVES.to(DEVICE)

    # spawn powers: (E_TYPES, P); roll ranges from each enemy's spawn_rolls
    sp = torch.zeros(C.E_TYPES, P)
    C.CURL_RANGE = torch.zeros(C.E_TYPES, 2)
    C.SPAWN_AMT_RANGE = torch.zeros(C.E_TYPES, 2)
    for ei, e in enumerate(enemies):
        rolls = e.get("spawn_rolls", {})
        for rname, spec in rolls.items():
            rng = spec.get("range_by_ascension")
            lo, hi = _asc(rng, ascension) if rng else (spec["min"], spec["max"])
            if rname == "curl_up":
                C.CURL_RANGE[ei] = torch.tensor([float(lo), float(hi)])
            else:
                C.SPAWN_AMT_RANGE[ei] = torch.tensor([float(lo), float(hi)])
        for pw in e.get("powers_on_spawn", []):
            a = pw["amount"]
            if isinstance(a, dict) and "spawn_roll" in a:
                sp[ei, PW[pw["power"]]] = -1.0      # roll at spawn
            else:
                sp[ei, PW[pw["power"]]] = float(_asc(a, ascension))
    C.SPAWN_POWERS = sp.to(DEVICE)
    C.CURL_RANGE = C.CURL_RANGE.to(DEVICE)
    C.SPAWN_AMT_RANGE = C.SPAWN_AMT_RANGE.to(DEVICE)

    C.GUARD_TID = C.eid.get("guardian", -99)
    from . import ai
    C.AI = ai.compile_ai(C, enemies, ascension)


def _load_encounters(C, enc):
    C.ENCOUNTERS = {e["id"]: e for e in enc["encounters"]}
    C.POOLS = enc["pools"]
    C.SELECTION = enc["selection_rules"]
