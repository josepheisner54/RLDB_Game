"""Enemy AI. Each enemy type compiles to a uniform spec:

    initial: list[int]  -- move sequence played first (may be empty)
    then:    ("repeat", seq) | ("weighted", weights, max_consec)
    hp_interrupt: (frac, move) | None      -- slimes' split, checked at intent
    special: None|"lagavulin"|"guardian"|"looter"|"shield_gremlin"|"sentry"

Per-slot runtime registers (B, E, R): 0 mode(0=initial,1=then, specials reuse),
1 pos, 2 last_move, 3 consec, 4 aux (guardian shift counter / lagavulin sleep
turns / looter coin), 5 woke_flag (lagavulin: took HP damage this round).

Approximations: Looter's lunge/smoke branch = coin flip into one of its two
listed continuations; Guardian returns to offensive with the documented
counter increment and the 'after_return' cycle.
"""
import torch
from .vocab import AI_WEIGHTED, AI_SEQUENCE

NREG = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AiSpec:
    def __init__(self):
        self.initial = []
        self.then = ("repeat", [0])
        self.hp_interrupt = None
        self.special = None


def _mv(C, eid, name):
    return C.move_id[(eid, name)]


def compile_ai(C, enemies, ascension):
    from .loader import _asc
    specs = []
    for e in enemies:
        ai, eid = e["ai"], e["id"]
        s = AiSpec()
        t = ai["type"]
        for itr in ai.get("interrupts", []):
            assert "50_percent" in itr["when"]
            s.hp_interrupt = (0.5, _mv(C, eid, itr["set_move"]))

        if t == "weighted_history":
            if "first_move" in ai:                      # jaw worm
                s.initial = [_mv(C, eid, ai["first_move"])]
            w = _asc(ai.get("weights_by_ascension",
                            ai.get("weights", ai.get("weights_after_first"))), ascension)
            cons = _asc(ai.get("constraints_by_ascension",
                               {"0": ai.get("constraints", [])}), ascension)
            names = list(e["moves"].keys())
            weights = torch.zeros(len(names))
            maxc = torch.full((len(names),), 99.0)
            for mid, wt in w.items():
                weights[_mv(C, eid, mid)] = float(wt)
            for c in cons or []:
                maxc[_mv(C, eid, c["move"])] = float(c["max_consecutive"])
            s.then = ("weighted", weights, maxc)
        elif t in ("fixed", "sequence_by_ascension"):
            spec = ai if t == "fixed" else _asc({k: v for k, v in ai.items()
                                                 if k != "type"}, ascension)
            s.initial = [_mv(C, eid, m) for m in spec.get("initial", [])]
            if "sequence" in spec:                       # gremlins, cultist
                seq = [_mv(C, eid, m) for m in spec["sequence"]]
                if spec.get("repeat_last"):
                    s.initial, s.then = seq[:-1], ("repeat", [seq[-1]])
                else:                                    # repeat: true
                    s.then = ("repeat", seq)
            elif "repeat" in spec:
                s.then = ("repeat", [_mv(C, eid, m) for m in spec["repeat"]])
            elif "then" in spec:                       # Gremlin Nob A0 hybrid
                th = spec["then"]
                names = list(e["moves"].keys())
                weights = torch.zeros(len(names))
                maxc = torch.full((len(names),), 99.0)
                for mid, wt in th["weighted"].items():
                    weights[_mv(C, eid, mid)] = float(wt)
                for c in th.get("constraints", []):
                    maxc[_mv(C, eid, c["move"])] = float(c["max_consecutive"])
                s.then = ("weighted", weights, maxc)
            elif "start_once_then_repeat_without_bellow" in spec:
                s.initial = [_mv(C, eid, m) for m in spec["repeat"]]
                s.then = ("repeat", [_mv(C, eid, m)
                                     for m in spec["start_once_then_repeat_without_bellow"]])
            else:
                raise ValueError(f"{eid}: unknown sequence spec {spec}")
        elif t == "alternating":
            s.special = "alternating"
            fm = _asc(ai.get("first_move_by_ascension",
                             ai.get("first_move", "specified_by_encounter_slot")),
                      ascension)
            if fm == "specified_by_encounter_slot":
                s.first = ("slot", None)                 # sentry: spawner sets aux
            elif isinstance(fm, dict):                   # acid slime S: random first
                names = list(e["moves"].keys())
                w = torch.zeros(len(names))
                for mid, wt in fm["weighted"].items():
                    w[_mv(C, eid, mid)] = float(wt)
                s.first = ("weighted", w)
            else:
                s.first = ("fixed", _mv(C, eid, fm))
        elif t == "conditional":
            s.special = "shield_gremlin"
            s.protect = _mv(C, eid, "protect")
            s.solo = _mv(C, eid, "shield_bash")
        elif t == "state_machine" and "sequence" in ai:   # looter
            s.special = "looter"
            s.initial = []
            seq = ai["sequence"]
            fixed = [_mv(C, eid, m) for m in seq if isinstance(m, str)]
            branch = seq[-1]["weighted"]
            b1, b2 = list(branch.keys())
            s.branches = ([ _mv(C, eid, b1)] + [_mv(C, eid, m) for m in ai["branches"][b1]],
                          [_mv(C, eid, b2)] + [_mv(C, eid, m) for m in ai["branches"][b2]])
            s.fixed = fixed
        elif t == "state_machine" and ai.get("initial_state") == "asleep":
            st = ai["states"]
            s.special = "lagavulin"
            s.sleep = _mv(C, eid, st["asleep"]["move"])
            s.stunned = _mv(C, eid, st["asleep"]["on_damage_wake_next_move"])
            s.wake_turns = st["asleep"]["automatic_wake_after_enemy_turns"]
            s.awake = [_mv(C, eid, m) for m in st["awake"]["repeat"]]
        elif t == "state_machine" and "initial_move" in ai:  # red slaver
            s.special = "red_slaver"
            st = ai["states"]
            s.initial = [_mv(C, eid, ai["initial_move"])]
            pre = st["pre_entangle"]
            s.ent_pct = pre["each_turn_entangle_chance_percent"] / 100.0
            s.ent_move = _mv(C, eid, "entangle")
            s.pre_cycle = [_mv(C, eid, m)
                           for m in _asc(pre["otherwise_cycle_by_ascension"], ascension)]
            post = st["post_entangle"]
            names = list(e["moves"].keys())
            w = torch.zeros(len(names)); mc = torch.full((len(names),), 99.0)
            for mid, wt in post["weights"].items():
                w[_mv(C, eid, mid)] = float(wt)
            for c in _asc(post["constraints_by_ascension"], ascension):
                mc[_mv(C, eid, c["move"])] = float(c["max_consecutive"])
            s.post = (w, mc)
        elif t == "state_machine" and "state_data" in ai:  # guardian
            sd = ai["state_data"]
            s.special = "guardian"
            off = sd["offensive"]
            s.off_cycle = [_mv(C, eid, m) for m in off["initial_cycle"]]
            s.off_return = [_mv(C, eid, m) for m in off["after_return_from_defensive"]]
            s.shift0 = float(_asc(off["mode_shift_counter_by_ascension"], ascension))
            s.shift_inc = float(off["on_return_increment_counter"])
            s.def_seq = [_mv(C, eid, m) for m in sd["defensive"]["sequence"]]
            s.shift_move = _mv(C, eid, off["interrupt"]["set_move"])
        elif t == "cycle_with_interrupt":                  # slime boss
            s.then = ("repeat", [_mv(C, eid, m) for m in ai["cycle"]])
        else:
            raise ValueError(f"{eid}: unknown ai {ai}")
        specs.append(s)
    return specs


@torch.no_grad()
def choose_moves(C, etype, alive, ehp, ehp_max, eblk, regs, turn):
    """Select each living enemy's move for this round (telegraphed).
    Returns global move indices (B, E); updates regs in place."""
    B, E = etype.shape
    move = torch.zeros(B, E, dtype=torch.long, device=DEVICE)
    for ti, spec in enumerate(C.AI):
        m0 = (etype == ti) & (alive > 0)      # FULL type mask: final writes use this
        m = m0                                 # branches may reduce m as they consume rows
        if not m0.any():
            continue
        mode, pos = regs[..., 0], regs[..., 1]
        last, consec, aux = regs[..., 2], regs[..., 3], regs[..., 4]
        local = torch.zeros(B, E, dtype=torch.long, device=DEVICE)

        if spec.special == "lagavulin":
            asleep = m & (mode == 0)
            aux[asleep] += 1
            woke_dmg = asleep & (regs[..., 5] > 0)
            timeout = asleep & (aux > spec.wake_turns) & ~woke_dmg
            local[woke_dmg] = spec.stunned
            mode[woke_dmg | timeout] = 1
            pos[woke_dmg] = 0
            still = asleep & ~woke_dmg & ~timeout
            local[still] = spec.sleep
            awake = m & (mode == 1) & ~woke_dmg
            if timeout.any():
                local[timeout] = spec.awake[0]
                pos[timeout] = 1
                awake = awake & ~timeout
            for k, mv in enumerate(spec.awake):
                sel = awake & (pos % len(spec.awake) == k)
                local[sel] = mv
            pos[awake] += 1
        elif spec.special == "guardian":
            off = m & (mode == 0)
            shift = off & (aux <= 0)
            local[shift] = spec.shift_move
            eblk[shift] += 20.0
            mode[shift] = 1; pos[shift] = 0
            stay = off & ~shift
            first = mode.new_zeros(mode.shape).bool()
            for k in range(4):
                cyc = spec.off_cycle if True else spec.off_return
                sel = stay & (pos % 4 == k)
                ret = sel & (regs[..., 5] > 1)         # returned-from-defensive flag
                local[sel] = spec.off_cycle[k]
                local[ret] = spec.off_return[k]
            pos[stay] += 1
            dfn = m & (mode == 1) & ~shift
            for k, mv in enumerate(spec.def_seq[1:], 1):
                sel = dfn & (pos == k)
                local[sel] = mv
            done = dfn & (pos >= len(spec.def_seq) - 1)
            pos[dfn] += 1
            mode[done] = 0; pos[done] = 0
            aux[done] = spec.shift0 + spec.shift_inc
            regs[..., 5][done] = 2                     # use off_return cycle
        elif spec.special == "looter":
            nf = len(spec.fixed)
            in_fixed = m & (pos < nf)
            for k, mv in enumerate(spec.fixed):
                local[in_fixed & (pos == k)] = mv
            br = m & (pos >= nf)
            fresh = br & (aux == 0)
            aux[fresh] = torch.where(torch.rand_like(aux[fresh]) < 0.5,
                                     torch.ones_like(aux[fresh]),
                                     2 * torch.ones_like(aux[fresh]))
            for bi, seq in enumerate(spec.branches, 1):
                for k, mv in enumerate(seq):
                    sel = br & (aux == bi) & (pos - nf == k)
                    local[sel] = mv
                over = br & (aux == bi) & (pos - nf >= len(seq))
                local[over] = seq[-1]                  # keep escaping
            pos[m] += 1
        elif spec.special == "shield_gremlin":
            others = ((alive > 0).sum(-1, keepdim=True) - 1).expand(B, E)
            local[m & (others > 0)] = spec.protect
            local[m & (others <= 0)] = spec.solo
        elif spec.special == "alternating":
            fresh = m & (aux == 0)
            if fresh.any():
                kind_f = spec.first[0]
                if kind_f == "weighted":
                    w = spec.first[1].to(DEVICE).expand(int(fresh.sum()), -1)
                    aux[fresh] = torch.multinomial(w.clamp(min=1e-8), 1).squeeze(-1).float() + 1
                elif kind_f == "fixed":
                    aux[fresh] = spec.first[1] + 1
                else:
                    aux[fresh] = 1.0                     # sentry default; spawner overrides
            local[m] = ((aux - 1).long() + pos.long())[m] % 2
            pos[m] += 1
        elif spec.special == "red_slaver":
            first = m & (pos == 0)
            local[first] = spec.initial[0]
            pre = m & (mode == 0) & ~first
            if pre.any():
                roll = torch.rand_like(aux) < spec.ent_pct
                ent = pre & roll
                local[ent] = spec.ent_move
                mode[ent] = 1
                cyc = pre & ~ent
                base = (pos - 1).long()
                for k, mv in enumerate(spec.pre_cycle):
                    local[cyc & (base % len(spec.pre_cycle) == k)] = mv
            post = m & (mode == 1) & ~first
            if post.any():
                w0, mc = spec.post
                w = w0.to(DEVICE).unsqueeze(0).unsqueeze(0).expand(B, E, -1).clone()
                blocked = (consec.unsqueeze(-1) >=
                           mc.to(DEVICE).gather(0, last.long().clamp(min=0, max=w.shape[-1] - 1).view(-1))
                           .view(B, E, 1)) &                           torch.nn.functional.one_hot(last.long().clamp(min=0, max=w.shape[-1] - 1),
                                                      w.shape[-1]).bool()
                w = w.masked_fill(blocked, 0.0)
                idx = torch.multinomial(w.view(B * E, -1).clamp(min=1e-8), 1).view(B, E)
                local = torch.where(post, idx, local)
            pos[m] += 1
        else:
            if spec.hp_interrupt is not None:
                frac, mv = spec.hp_interrupt
                split = m & (ehp <= frac * ehp_max) & (mode < 2)
                local[split] = mv
                mode[split] = 2                        # split fires once
                m = m & ~split
            if spec.initial:
                ini = m & (pos < len(spec.initial))
                for k, mv in enumerate(spec.initial):
                    local[ini & (pos == k)] = mv
                pos[ini] += 1
                m = m & ~ini
            kind = spec.then[0]
            if kind == "repeat":
                seq = spec.then[1]
                base = pos - len(spec.initial)
                for k, mv in enumerate(seq):
                    local[m & (base % len(seq) == k)] = mv
                pos[m] += 1
            else:
                _, weights, maxc = spec.then
                w = weights.to(DEVICE).unsqueeze(0).unsqueeze(0).expand(B, E, -1).clone()
                blocked = (consec.unsqueeze(-1) >=
                           maxc.to(DEVICE).gather(0, last.long().clamp(min=0, max=w.shape[-1] - 1).view(-1))
                           .view(B, E, 1)) & \
                          (torch.nn.functional.one_hot(last.long().clamp(min=0, max=w.shape[-1] - 1),
                                                       w.shape[-1]).bool())
                w = w.masked_fill(blocked, 0.0)
                idx = torch.multinomial(w.view(B * E, -1).clamp(min=1e-8), 1).view(B, E)
                local = torch.where(m, idx, local)
        rep = (local == last.long()) & m0
        consec[rep] += 1
        consec[m0 & ~rep] = 1
        last[m0] = local[m0].float()
        move = torch.where(m0, C.MOVE_BASE[ti] + local, move)
        if spec.special == "lagavulin":
            regs[..., 5][m0] = 0.0             # wake flag consumed each round
    return move
