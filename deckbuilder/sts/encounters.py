"""Spawn encounters into batched enemy slot tensors.

Kinds: fixed, generated (per-slot uniform), choose_uniform_formation,
sample_without_replacement (gremlin gang, via Gumbel top-k over the bag).
"""
import torch
from .vocab import E_MAX, PW
from .ai import NREG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def blank_slots(B):
    return dict(
        etype=torch.full((B, E_MAX), -1, dtype=torch.long, device=DEVICE),
        ehp=torch.zeros(B, E_MAX, device=DEVICE),
        emax=torch.zeros(B, E_MAX, device=DEVICE),
        eblk=torch.zeros(B, E_MAX, device=DEVICE),
        epow=torch.zeros(B, E_MAX, len(PW), device=DEVICE),
        regs=torch.zeros(B, E_MAX, NREG, device=DEVICE),
        stolen=torch.zeros(B, E_MAX, device=DEVICE),
        spawn_amt=torch.zeros(B, E_MAX, device=DEVICE),
        hp_budget=torch.zeros(B, device=DEVICE),
    )


def init_slot(C, slots, rows, slot_idx, etype_ids, first_move=None):
    """Fill slot `slot_idx` for batch rows `rows` with enemy types (len(rows),)."""
    lo = C.E_HP[etype_ids, 0]; hi = C.E_HP[etype_ids, 1]
    hp = torch.floor(torch.rand(len(rows), device=DEVICE) * (hi - lo + 1)) + lo
    slots["etype"][rows, slot_idx] = etype_ids
    slots["ehp"][rows, slot_idx] = hp
    slots["emax"][rows, slot_idx] = hp
    slots["eblk"][rows, slot_idx] = 0.0
    pw = C.SPAWN_POWERS[etype_ids].clone()
    roll = pw[:, PW["curl_up"]] < 0
    if roll.any():
        cr = C.CURL_RANGE[etype_ids[roll]]
        pw[roll, PW["curl_up"]] = torch.floor(
            torch.rand(int(roll.sum()), device=DEVICE) * (cr[:, 1] - cr[:, 0] + 1)) + cr[:, 0]
    slots["epow"][rows, slot_idx] = pw
    slots["regs"][rows, slot_idx] = 0.0
    rng = C.SPAWN_AMT_RANGE[etype_ids]
    slots["spawn_amt"][rows, slot_idx] = torch.floor(
        torch.rand(len(rows), device=DEVICE) * (rng[:, 1] - rng[:, 0] + 1)) + rng[:, 0]
    if first_move is not None:
        slots["regs"][rows, slot_idx, 4] = float(first_move + 1)   # alternating aux
    slots["stolen"][rows, slot_idx] = 0.0


def spawn_encounter(C, enc_ids, B):
    """enc_ids: list[str] length B (may repeat). Returns fresh slot tensors."""
    slots = blank_slots(B)
    ids = list(enc_ids)
    for enc_id in set(ids):
        rows = torch.tensor([i for i, x in enumerate(ids) if x == enc_id],
                            device=DEVICE, dtype=torch.long)
        e = C.ENCOUNTERS[enc_id]
        kind = e["kind"]
        if kind == "fixed":
            for si, spec in enumerate(e["enemies"]):
                et = torch.full((len(rows),), C.eid[spec["enemy"]],
                                dtype=torch.long, device=DEVICE)
                fm = None
                if "ai_parameters" in spec:
                    fm = C.move_id[(spec["enemy"], spec["ai_parameters"]["first_move"])]
                init_slot(C, slots, rows, si, et, first_move=fm)
                slots["hp_budget"][rows] += slots["ehp"][rows, si]
        elif kind == "generated":
            for si, spec in enumerate(e["slots"]):
                opts = torch.tensor([C.eid[n] for n in spec["choose_uniform"]],
                                    device=DEVICE)
                et = opts[torch.randint(0, len(opts), (len(rows),), device=DEVICE)]
                init_slot(C, slots, rows, si, et)
                slots["hp_budget"][rows] += slots["ehp"][rows, si]
        elif kind == "choose_uniform_formation":
            forms = e["formations"]
            pick = torch.randint(0, len(forms), (len(rows),), device=DEVICE)
            for fi, form in enumerate(forms):
                sub = rows[pick == fi]
                if len(sub) == 0:
                    continue
                for si, spec in enumerate(form):
                    et = torch.full((len(sub),), C.eid[spec["enemy"]],
                                    dtype=torch.long, device=DEVICE)
                    init_slot(C, slots, sub, si, et)
                    slots["hp_budget"][sub] += slots["ehp"][sub, si]
        elif kind == "sample_without_replacement":
            bag = torch.tensor([C.eid[n] for n in e["bag"]], device=DEVICE)
            g = torch.rand(len(rows), len(bag), device=DEVICE)
            order = g.argsort(dim=1)[:, :e["count"]]
            for si in range(e["count"]):
                init_slot(C, slots, rows, si, bag[order[:, si]])
                slots["hp_budget"][rows] += slots["ehp"][rows, si]
        else:
            raise ValueError(f"unknown encounter kind {kind}")
    return slots
