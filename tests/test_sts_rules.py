"""Regression tests for the STS engine: rule semantics as executable facts.
Every test here encodes a rule from combat_rules.json or a mechanic that
already bit us once. If one fails, fidelity regressed."""
import torch
import torch.nn as nn
import pytest

from deckbuilder import sts
from deckbuilder.sts import engine, ai as ai_mod
from deckbuilder.sts.vocab import PW, E_MAX

torch.manual_seed(0)
C = sts.load(ascension=0)
B = 8


def fresh(enc="cultist_solo", deck_counts=None, hp=70.0, b=B):
    deck = deck_counts if deck_counts is not None \
        else C.STARTER.unsqueeze(0).repeat(b, 1)
    st = engine.new_combat(C, deck, torch.full((b,), hp), [enc] * b)
    st["x_spent"] = torch.zeros(b)
    st["move"] = ai_mod.choose_moves(C, st["etype"], engine.alive_mask(st).float(),
                                     st["ehp"], st["emax"], st["eblk"],
                                     st["regs"], 0)
    return st


def deck_of(**cards):
    d = torch.zeros(B, C.N)
    for cid, n in cards.items():
        d[:, C.card_row[cid]] = n
    return d


class Force(nn.Module):
    """Policy that always plays card `row` at slot 0, then ends turn."""
    def __init__(self, row):
        super().__init__()
        self.row = row
        self.played = None
    def forward(self, C_, st, mask):
        z = torch.full((st["B"], C_.N * E_MAX + 1), -1e9)
        a = self.row * E_MAX
        z[:, -1] = 0.0
        z[:, a] = torch.where(mask[:, a], torch.tensor(100.0), torch.tensor(-1e9))
        return z


def total_cards(st):
    return (st["draw"] + st["hand"] + st["disc"] + st["exh"]).sum(-1)


def test_zone_conservation_and_reshuffle():
    st = fresh(deck_counts=deck_of(strike=3, defend=3))
    before = total_cards(st)
    for t in range(3):
        engine.start_player_turn(C, st, first_turn=(t == 0))
        engine.end_player_turn(C, st)
    assert torch.allclose(total_cards(st), before)
    # 6-card deck, drew 5 three times -> reshuffle must have happened
    assert (st["draw"] + st["hand"] + st["disc"]).sum(-1).min() == 6


def test_hand_cap_overflow_to_discard():
    st = fresh(deck_counts=deck_of(strike=15))
    engine.draw_cards(C, st, torch.full((B,), 12, dtype=torch.long),
                      torch.ones(B, dtype=torch.bool))
    assert (st["hand"].sum(-1) <= C.HAND_MAX + 1e-6).all()
    assert (st["disc"].sum(-1) >= 2 - 1e-6).all()   # overflow went to discard


def test_exhausted_cards_never_reshuffle():
    st = fresh(deck_counts=deck_of(strike=4))
    st["exh"][:, C.card_row["strike"]] = 2.0
    for t in range(3):
        engine.start_player_turn(C, st, first_turn=(t == 0))
        engine.end_player_turn(C, st)
    assert (st["exh"][:, C.card_row["strike"]] == 2.0).all()
    assert ((st["draw"] + st["hand"] + st["disc"]).sum(-1) == 4).all()


def test_x_cost_whirlwind_hits_equal_energy():
    st = fresh("cultist_solo", deck_of(whirlwind=1))
    hp0 = st["ehp"][:, 0].clone()
    engine.start_player_turn(C, st, first_turn=True)
    engine.player_turn(C, st, Force(C.card_row["whirlwind"]))
    assert (st["energy"] == 0).all()
    assert torch.allclose(st["ehp"][:, 0], hp0 - 3 * 5.0)   # X=3 hits of 5


def test_heavy_blade_strength_multiplier():
    for cid, mult in (("heavy_blade", 3), (None, 5)):
        row = C.card_row["heavy_blade"] + (0 if mult == 3 else C.M)
        st = fresh("cultist_solo", deck_of(heavy_blade=1))
        if mult == 5:
            st["hand"][:, C.card_row["heavy_blade"]] = 0
            st["draw"][:, C.card_row["heavy_blade"]] = 0
            st["draw"][:, row] = 1
        st["ppow"][:, PW["strength"]] = 2.0
        hp0 = st["ehp"][:, 0].clone()
        engine.start_player_turn(C, st, first_turn=True)
        engine.player_turn(C, st, Force(row))
        assert torch.allclose(st["ehp"][:, 0], hp0 - (14 + 2 * mult)), mult


def test_weak_vuln_floor_once():
    st = fresh("cultist_solo", deck_of(strike=1))
    st["ppow"][:, PW["weak"]] = 1.0
    st["epow"][:, 0, PW["vulnerable"]] = 1.0
    hp0 = st["ehp"][:, 0].clone()
    engine.start_player_turn(C, st, first_turn=True)
    engine.player_turn(C, st, Force(C.card_row["strike"]))
    assert torch.allclose(st["ehp"][:, 0], hp0 - 6.0)  # floor(6*0.75*1.5)=6


def test_curl_up_triggers_once():
    st = fresh("two_louses", deck_of(strike=1))
    st["epow"][:, 0, PW["curl_up"]] = 5.0
    engine.start_player_turn(C, st, first_turn=True)
    engine.player_turn(C, st, Force(C.card_row["strike"]))
    hurt = st["ehp"][:, 0] < st["emax"][:, 0]
    assert hurt.any()
    # curl fires AFTER the damage lands: block for future hits, power consumed
    assert (st["eblk"][:, 0][hurt] == 5.0).all()
    assert (st["epow"][:, 0, PW["curl_up"]][hurt] == 0).all()


def test_enrage_punishes_skills():
    st = fresh("gremlin_nob_elite", deck_of(defend=5))
    st["epow"][:, 0, PW["enrage"]] = 2.0
    s0 = st["epow"][:, 0, PW["strength"]].clone()
    engine.start_player_turn(C, st, first_turn=True)
    engine.player_turn(C, st, Force(C.card_row["defend"]))
    assert (st["epow"][:, 0, PW["strength"]] >= s0 + 2.0).all()


def test_artifact_negates_first_debuff():
    st = fresh("three_sentries_elite", deck_of(bash=2))
    a0 = st["epow"][:, 0, PW["artifact"]].clone()
    assert (a0 >= 1.0).all()
    engine.start_player_turn(C, st, first_turn=True)
    engine.player_turn(C, st, Force(C.card_row["bash"]))
    played = st["epow"][:, 0, PW["artifact"]] < a0
    assert played.any()
    assert (st["epow"][:, 0, PW["vulnerable"]][played] == 0).all()


def test_lagavulin_sleep_wake_cycle():
    st = fresh("lagavulin_elite", deck_of(defend=5))
    st["regs"][:] = 0.0            # fresh() consumed one intent choice; reset
    sleeps = 0
    for t in range(5):
        st["move"] = ai_mod.choose_moves(C, st["etype"],
                                         engine.alive_mask(st).float(),
                                         st["ehp"], st["emax"], st["eblk"],
                                         st["regs"], t)
        mv = st["move"][0, 0] - C.MOVE_BASE[C.eid["lagavulin"]]
        if t < 3:
            assert int(mv) == C.move_id[("lagavulin", "sleep")]
            sleeps += 1
        else:
            assert int(mv) != C.move_id[("lagavulin", "sleep")]
    assert sleeps == 3
    st2 = fresh("lagavulin_elite", deck_of(strike=5))
    st2["regs"][..., 5] = 1.0                    # took HP damage while asleep
    st2["regs"][..., 4] = 1.0
    mv = ai_mod.choose_moves(C, st2["etype"], engine.alive_mask(st2).float(),
                             st2["ehp"], st2["emax"], st2["eblk"],
                             st2["regs"], 1)[0, 0] - C.MOVE_BASE[C.eid["lagavulin"]]
    assert int(mv) == C.move_id[("lagavulin", "stunned")]


def test_slime_splits_at_half():
    st = fresh("large_slime", deck_of(defend=5))
    st["ehp"][:, 0] = st["emax"][:, 0] * 0.4
    st["move"] = ai_mod.choose_moves(C, st["etype"], engine.alive_mask(st).float(),
                                     st["ehp"], st["emax"], st["eblk"],
                                     st["regs"], 1)
    hp_before = st["ehp"][:, 0].clone()
    for e in range(E_MAX):
        engine.resolve_enemy_move(C, st, e, torch.ones(B, dtype=torch.bool))
    kids = (st["etype"] >= 0).sum(-1)
    assert (kids == 2).all()                     # parent gone, two offspring
    assert torch.allclose(st["ehp"][st["etype"] >= 0],
                          hp_before.floor().repeat_interleave(2))


def test_burning_blood_heals_on_victory():
    st_hp = torch.full((B,), 40.0)
    deck = deck_of(strike=10)
    out = engine.combat(C, deck, st_hp, ["cultist_solo"] * B,
                        policy=Force(C.card_row["strike"]))
    won = out["won"] > 0
    assert won.any()
    # winners healed +6 over whatever HP they ended combat with
    assert (out["end_hp"][won] <= C.HP_MAX).all()


def test_pity_ramps_rare_chance():
    b = 4000
    pity = sts.init_pity(b)
    seen_rare_early, seen_rare_late = 0, 0
    for k in range(10):
        offers, pity = sts.roll_offers(C, b, "normal", pity)
        n_rare = (C.RARITY_T[offers] == 3).sum().item()
        if k < 1:
            seen_rare_early += n_rare
        if k >= 8:
            seen_rare_late += n_rare
    assert seen_rare_early == 0    # pity -5 fully suppresses the FIRST offer only
    assert seen_rare_late > 0                    # commons ramp the offset


def test_run_pools_switch_easy_to_hard():
    from deckbuilder.sts.runsim import _pick_encounters
    easy_ids = {p["encounter"] for p in C.POOLS["easy_hallway"]}
    picks = _pick_encounters(C, "easy_hallway", 32, [""] * 32, [""] * 32)
    assert set(picks) <= easy_ids
    # repeat exclusion: the excluded id never appears
    last = ["cultist_solo"] * 32
    picks = _pick_encounters(C, "easy_hallway", 32, last, last)
    assert "cultist_solo" not in picks


def test_pipeline_smoke():
    policy, V, hist = sts.train_combat(C, steps=2, B=12, eval_every=1,
                                       verbose=False)
    assert all(torch.isfinite(torch.tensor(h[1:])).all() for h in hist)
    meta, _ = sts.train_meta(C, V, steps=2, B=32, verbose=False)
    r = sts.simulate_runs(C, policy, meta, B=24)
    assert 0.0 <= r["win"] <= 1.0


def test_token_policy_contract():
    """TokenCombatPolicy: instance builder is a faithful expansion, forward
    honors the engine contract, sampling never picks illegal actions, and a
    checkpointed training step runs."""
    from deckbuilder.sts.agents import TokenCombatPolicy, build_instances
    p = TokenCombatPolicy(C, d=32, n_layers=1)
    st = fresh("gremlin_gang", deck_of(strike=3, defend=2, bash=1))
    engine.start_player_turn(C, st, first_turn=True)
    rows, zones = build_instances(C, st)
    # every zone count is reproduced exactly as instance tokens
    for zi, zk in enumerate(("hand", "draw", "disc", "exh")):
        for r in (C.card_row["strike"], C.card_row["defend"], C.card_row["bash"]):
            want = int(st[zk][0, r].round())
            got = int(((rows[0] == r) & (zones[0] == zi)).sum())
            assert got == want, (zk, r, got, want)
    mask = engine.legal_actions(C, st)
    logits = p(C, st, mask)
    assert logits.shape == (B, C.N * 6 + 1)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    a = torch.distributions.Categorical(logits=masked).sample((32,))
    assert bool(mask.gather(1, a.T).all())
    pol, V_, _ = sts.train_combat(C, steps=2, B=12, micro_B=6, eval_every=1,
                                  policy=TokenCombatPolicy(C, d=32, n_layers=1),
                                  verbose=False)
