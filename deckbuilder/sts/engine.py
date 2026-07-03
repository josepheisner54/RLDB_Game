"""Batched STS combat engine over compiled micro-programs.

Zones are count vectors over the 155 card rows (unordered; the two
"pile-order" cards act on random pile cards). All events execute as masked
batch ops so heterogeneous cards/enemies resolve in one pass.

Rules honored (see content/combat_rules.json): damage = (base + str*mult)
* 1.5 vuln * 0.75 weak, floored once, each hit separately vs block; card
block gets dexterity then frail; hand cap 10 with overflow-to-discard;
reshuffle discard->draw when empty; ethereal exhausts at end of turn; whole
hand discards at end of turn; block clears at owner's turn start (Barricade
exempt); statuses/powers per the 34-power vocabulary.
"""
import torch
import torch.nn.functional as F

from .vocab import (OP, PW, P, AMT_X, AMT_PLAYER_BLOCK, AMT_RAMPAGE,
                    AMT_SOURCE_HP, AMT_CAPTURED, AMT_DIVIDER, AMT_PERFECTED, AMT_SPAWN,
                    CARD_DYNAMIC_BURN, TGT_CHOSEN, TGT_PLAYER,
                    TGT_ALL_ENEMIES, TGT_RANDOM_EACH_HIT, TGT_SELF, TGT_ALLY,
                    CND_NONE, CND_TGT_VULN, CND_TGT_INTENT_ATK, CND_FATAL,
                    DEST_DRAW, DEST_DISCARD, DEST_HAND, CT_ATTACK, CT_SKILL,
                    CT_POWER, CT_STATUS, MAX_FX, E_MAX, PLAY_CAP, HIT_CAP,
                    TURN_CAP, INTENT_ATTACK)
from . import ai as ai_mod
from .encounters import spawn_encounter, init_slot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def new_combat(C, deck_counts, php, enc_ids):
    B = deck_counts.shape[0]
    st = spawn_encounter(C, enc_ids, B)
    st.update(
        php=php.clone(), pblock=torch.zeros(B, device=DEVICE),
        energy=torch.zeros(B, device=DEVICE),
        ppow=torch.zeros(B, P, device=DEVICE),
        draw=deck_counts.clone(), hand=torch.zeros(B, C.N, device=DEVICE),
        disc=torch.zeros(B, C.N, device=DEVICE),
        exh=torch.zeros(B, C.N, device=DEVICE),
        rampage=torch.full((B,), 8.0, device=DEVICE),
        captured=torch.zeros(B, device=DEVICE),
        inferno=torch.zeros(B, device=DEVICE),
        gold_lost=torch.zeros(B, device=DEVICE),
        move=torch.zeros(B, E_MAX, dtype=torch.long, device=DEVICE),
        turn=0, B=B,
    )
    return st


def alive_mask(st):
    return (st["etype"] >= 0) & (st["ehp"] > 0)


def player_alive(st):
    return st["php"] > 0


# ---------------------------------------------------------------- drawing
def draw_cards(C, st, n_each, active):
    """Draw one card at a time (reshuffles between draws, on-draw triggers)."""
    B = st["B"]
    no_draw = st["ppow"][:, PW["no_draw"]] > 0
    for _ in range(int(n_each.max().item()) if torch.is_tensor(n_each) else n_each):
        want = active & player_alive(st) & ~no_draw
        if torch.is_tensor(n_each):
            want = want & (n_each > 0)
        if not want.any():
            break
        empty = st["draw"].sum(-1) < 0.5
        reshuffle = want & empty & (st["disc"].sum(-1) > 0.5)
        st["draw"][reshuffle] += st["disc"][reshuffle]
        st["disc"][reshuffle] = 0.0
        can = want & (st["draw"].sum(-1) > 0.5)
        if can.any():
            idx = torch.multinomial(st["draw"][can].clamp(min=1e-9), 1).squeeze(-1)
            one = F.one_hot(idx, C.N).float()
            st["draw"][can] -= one
            full = st["hand"][can].sum(-1) >= C.HAND_MAX
            st["hand"][can] += one * (~full).unsqueeze(-1).float()
            st["disc"][can] += one * full.unsqueeze(-1).float()
            # on-draw triggers for statuses: Evolve (draw), Fire Breathing (dmg all)
            is_status = (C.CTYPE[idx] == CT_STATUS).float() * (~full).float()
            ev = st["ppow"][can][:, PW["evolve"]] * is_status
            fb = st["ppow"][can][:, PW["fire_breathing"]] * is_status
            if (fb > 0).any():
                dmg = torch.zeros(B, device=DEVICE)
                dmg[can] = fb
                _hit_enemies(C, st, dmg, alive_mask(st), source_player=True,
                             is_attack=False)
            if (ev > 0).any():
                sub = torch.zeros(B, device=DEVICE)
                sub[can] = ev
                draw_cards(C, st, sub.long(), sub > 0)
        if torch.is_tensor(n_each):
            n_each = n_each - want.long()


# ---------------------------------------------------------------- damage
def _floor(x):
    return torch.floor(x.clamp(min=0.0))


def _hit_enemies(C, st, per_hit, tmask, source_player=True, is_attack=True,
                 lag_types=None):
    """Apply one hit of `per_hit` (B,) to enemies selected by tmask (B,E).
    Handles vulnerable, block, Curl Up, Angry, Lagavulin wake, deaths."""
    vuln = 1.0 + 0.5 * (st["epow"][..., PW["vulnerable"]] > 0).float()
    dmg = _floor(per_hit.unsqueeze(-1) * vuln) * tmask.float()
    absorbed = torch.minimum(dmg, st["eblk"])
    st["eblk"] -= absorbed
    hp_dmg = dmg - absorbed
    took = hp_dmg > 0
    st["ehp"] -= hp_dmg
    if is_attack and source_player:
        curl = took & (st["epow"][..., PW["curl_up"]] > 0)
        st["eblk"][curl] += st["epow"][..., PW["curl_up"]][curl]
        st["epow"][..., PW["curl_up"]][curl] = 0.0
        angry = took & (st["epow"][..., PW["angry"]] > 0)
        st["epow"][..., PW["strength"]][angry] += st["epow"][..., PW["angry"]][angry]
    # lagavulin wake flag on HP damage (max: don't stomp guardian's flag=2)
    st["regs"][..., 5] = torch.maximum(st["regs"][..., 5], took.float())
    # guardian mode-shift counter counts down with HP damage taken
    gm = took & (st["etype"] == getattr(C, "GUARD_TID", -99))
    st["regs"][..., 4] = torch.where(gm, st["regs"][..., 4] - hp_dmg,
                                     st["regs"][..., 4])
    _handle_deaths(C, st)


def _handle_deaths(C, st):
    dead = (st["etype"] >= 0) & (st["ehp"] <= 0)
    if not dead.any():
        return
    spore = dead & (st["epow"][..., PW["spore_cloud"]] > 0)
    if spore.any():
        amt = (st["epow"][..., PW["spore_cloud"]] * spore.float()).sum(-1)
        st["ppow"][:, PW["vulnerable"]] += amt
        st["epow"][..., PW["spore_cloud"]][spore] = 0.0
    st["stolen"][dead] = 0.0            # gold recovered on kill
    st["etype"][dead] = -1


def _hit_player(C, st, per_hit, source_mask=None, from_card=False):
    """One hit to the player. source_mask (B,E): the attacking enemy (for
    Flame Barrier reflection); from_card=True routes Rupture."""
    vuln = 1.0 + 0.5 * (st["ppow"][:, PW["vulnerable"]] > 0).float()
    dmg = _floor(per_hit * vuln)
    absorbed = torch.minimum(dmg, st["pblock"])
    st["pblock"] -= absorbed
    hp = dmg - absorbed
    st["php"] -= hp
    if from_card:
        rup = (hp > 0) & (st["ppow"][:, PW["rupture"]] > 0)
        st["ppow"][:, PW["strength"]][rup] += st["ppow"][:, PW["rupture"]][rup]
    if source_mask is not None:
        fb = st["ppow"][:, PW["flame_barrier"]]
        if (fb > 0).any():
            _hit_enemies(C, st, fb, source_mask & (fb > 0).unsqueeze(-1),
                         source_player=True, is_attack=False)


def _lose_hp(C, st, amount, active, from_card=True):
    hp = _floor(amount) * active.float()
    st["php"] -= hp
    if from_card:
        rup = (hp > 0) & (st["ppow"][:, PW["rupture"]] > 0)
        st["ppow"][:, PW["strength"]][rup] += st["ppow"][:, PW["rupture"]][rup]


def _player_gain_block(C, st, amount, active, from_card=True):
    b = amount.clone()
    if from_card:
        b = b + st["ppow"][:, PW["dexterity"]]
        b = b * torch.where(st["ppow"][:, PW["frail"]] > 0,
                            torch.tensor(0.75, device=DEVICE),
                            torch.tensor(1.0, device=DEVICE))
    b = _floor(b) * active.float()
    st["pblock"] += b
    jug = st["ppow"][:, PW["juggernaut"]]
    fire = (b > 0) & (jug > 0)
    if fire.any():
        am = alive_mask(st)
        r = torch.rand_like(st["ehp"]).masked_fill(~am, -1.0)
        pick = F.one_hot(r.argmax(-1), E_MAX).bool() & am
        _hit_enemies(C, st, jug * fire.float(), pick & fire.unsqueeze(-1),
                     source_player=True, is_attack=False)


# ---------------------------------------------------------------- resolver
def _select_targets(st, tcode, chosen, hit_i):
    """(B,) tcode + chosen slot -> enemy mask (B,E). Player codes -> None."""
    B = st["B"]
    am = alive_mask(st)
    tm = torch.zeros(B, E_MAX, dtype=torch.bool, device=DEVICE)
    is_ch = tcode == TGT_CHOSEN
    if is_ch.any():
        tm |= F.one_hot(chosen.clamp(min=0), E_MAX).bool() & is_ch.unsqueeze(-1)
    is_all = tcode == TGT_ALL_ENEMIES
    tm |= am & is_all.unsqueeze(-1)
    is_rnd = tcode == TGT_RANDOM_EACH_HIT
    if is_rnd.any():
        r = (torch.rand(B, E_MAX, device=DEVICE) + hit_i * 0.0).masked_fill(~am, -1.0)
        tm |= F.one_hot(r.argmax(-1), E_MAX).bool() & is_rnd.unsqueeze(-1) & am
    return tm & am


def resolve_player_card(C, st, row, chosen, active, nested=False):
    """Resolve one played card row (B,) with chosen enemy slot (B,).
    Returns exhaust_flags (B,) from exhaust_self ops."""
    B = st["B"]
    prog = C.PROG[row]                                  # (B, MAX_FX, NPARAM)
    exhaust_flag = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    fatal_credit = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    pre_alive = alive_mask(st).clone()
    for k in range(MAX_FX):
        opk = prog[:, k, 0].long()
        act = active & (opk != OP["pad"]) & player_alive(st)
        if not act.any():
            continue
        cond = prog[:, k, 6].long()
        # condition gates
        tvuln = st["epow"][..., PW["vulnerable"]].gather(
            1, chosen.clamp(min=0).unsqueeze(1)).squeeze(1) > 0
        tint = C.MOVE_INTENT[st["move"].gather(
            1, chosen.clamp(min=0).unsqueeze(1)).squeeze(1)] == INTENT_ATTACK
        act = act & ~((cond == CND_TGT_VULN) & ~tvuln)
        act = act & ~((cond == CND_TGT_INTENT_ATK) & ~tint)
        act = act & ~((cond == CND_FATAL) & ~fatal_credit)
        if not act.any():
            continue
        amt = prog[:, k, 1].clone()
        amt = torch.where(amt == AMT_X, st["x_spent"], amt)
        amt = torch.where(amt == AMT_PLAYER_BLOCK, st["pblock"], amt)
        amt = torch.where(amt == AMT_RAMPAGE, st["rampage"], amt)
        amt = torch.where(amt == AMT_CAPTURED, st["captured"], amt)
        if (prog[:, k, 1] == AMT_PERFECTED).any():
            strikes = ((st["hand"] + st["draw"] + st["disc"])
                       * C.STRIKE_MASK).sum(-1) + 1.0   # +1: the card itself
            amt = torch.where(prog[:, k, 1] == AMT_PERFECTED,
                              prog[:, k, 7] + prog[:, k, 5] * strikes, amt)
        hits = prog[:, k, 2].clone()
        hits = torch.where(hits == AMT_X, st["x_spent"], hits).clamp(min=0)
        tcode = prog[:, k, 4].long()

        o = OP
        m = act & (opk == o["damage"])
        if m.any():
            strength = st["ppow"][:, PW["strength"]] * prog[:, k, 3]
            weak = torch.where(st["ppow"][:, PW["weak"]] > 0,
                               torch.tensor(0.75, device=DEVICE),
                               torch.tensor(1.0, device=DEVICE))
            per_hit = (amt + strength) * weak
            for h in range(HIT_CAP):
                hm = m & (hits > h)
                if not hm.any():
                    break
                tm = _select_targets(st, tcode, chosen, h)
                _hit_enemies(C, st, per_hit * hm.float(),
                             tm & hm.unsqueeze(-1), source_player=True)
        m = act & (opk == o["damage_self"])
        if m.any():
            _hit_player(C, st, amt * m.float())
        m = act & (opk == o["gain_block"])
        if m.any():
            _player_gain_block(C, st, amt, m)
        m = act & (opk == o["apply_power"])
        if m.any():
            pid = prog[:, k, 3].long()
            to_player = m & ((tcode == TGT_PLAYER) | (tcode == TGT_SELF))
            for p_ in pid[to_player].unique():
                sel = to_player & (pid == p_)
                st["ppow"][:, p_][sel] += amt[sel]
            to_en = m & ~to_player
            if to_en.any():
                tm = _select_targets(st, tcode, chosen, 0) & to_en.unsqueeze(-1)
                art = st["epow"][..., PW["artifact"]] > 0
                blockable = tm & art
                st["epow"][..., PW["artifact"]][blockable] -= 1.0
                tm = tm & ~blockable
                for p_ in pid[to_en].unique():
                    sel = tm & (pid == p_).unsqueeze(-1)
                    st["epow"][..., p_][sel] += amt.unsqueeze(-1).expand_as(sel)[sel]
        m = act & (opk == o["draw"])
        if m.any():
            draw_cards(C, st, (amt * m.float()).long(), m)
        m = act & (opk == o["gain_energy"])
        st["energy"] += amt * m.float()
        m = act & (opk == o["lose_hp"])
        if m.any():
            _lose_hp(C, st, amt * m.float(), m)
        m = act & (opk == o["heal"])
        st["php"] = torch.minimum(st["php"] + amt * m.float(),
                                  torch.full_like(st["php"], C.HP_MAX))
        m = act & (opk == o["gain_max_hp"])
        st["php"] += amt * m.float()                    # run-level max hp handled upstream
        m = act & (opk == o["create_card"])
        if m.any():
            crow = prog[:, k, 3].long()
            dynb = crow == CARD_DYNAMIC_BURN
            crow = torch.where(dynb & (st["inferno"] > 0),
                               torch.full_like(crow, C.BURN_PLUS_ROW),
                               torch.where(dynb, torch.full_like(crow, C.BURN_ROW), crow))
            dest = prog[:, k, 5].long()
            onehot = F.one_hot(crow.clamp(min=0), C.N).float() * (amt * m.float()).unsqueeze(-1)
            st["draw"] += onehot * (dest == DEST_DRAW).float().unsqueeze(-1)
            st["disc"] += onehot * (dest == DEST_DISCARD).float().unsqueeze(-1)
            room = (C.HAND_MAX - st["hand"].sum(-1)).clamp(min=0)
            to_hand = onehot * (dest == DEST_HAND).float().unsqueeze(-1)
            take = torch.minimum(to_hand.sum(-1), room)
            scale = torch.where(to_hand.sum(-1) > 0, take / to_hand.sum(-1).clamp(min=1e-9),
                                torch.zeros_like(take))
            st["hand"] += to_hand * scale.unsqueeze(-1)
            st["disc"] += to_hand * (1 - scale).unsqueeze(-1)
        m = act & (opk == o["exhaust_self"])
        exhaust_flag |= m
        m = act & (opk == o["exhaust_random"])
        if m.any():
            cnt = torch.zeros(B, device=DEVICE)
            for _ in range(int(amt[m].max().item())):
                have = m & (st["hand"].sum(-1) > 0.5) & (amt > cnt)
                if not have.any():
                    break
                idx = torch.multinomial(st["hand"][have].clamp(min=1e-9), 1).squeeze(-1)
                one = F.one_hot(idx, C.N).float()
                st["hand"][have] -= one
                st["exh"][have] += one
                _on_exhaust(C, st, have, 1.0)
                cnt[have] += 1
            st["captured"] = torch.where(m, cnt, st["captured"])
        m = act & (opk == o["exhaust_hand"])
        if m.any():
            filt = prog[:, k, 5]
            keep_attacks = (filt == 1.0)
            sel = st["hand"].clone()
            sel[keep_attacks.unsqueeze(-1) & C.ATTACK_ROWS.unsqueeze(0)] = 0.0
            sel = sel * m.float().unsqueeze(-1)
            n_ex = sel.sum(-1)
            st["hand"] -= sel
            st["exh"] += sel
            _on_exhaust(C, st, m & (n_ex > 0), n_ex)
            st["captured"] = torch.where(m, n_ex, st["captured"])
        m = act & (opk == o["block_per_captured"])
        if m.any():
            _player_gain_block(C, st, amt * st["captured"], m)
        m = act & (opk == o["move_discard_to_draw"])
        if m.any():
            have = m & (st["disc"].sum(-1) > 0.5)
            if have.any():
                idx = torch.multinomial(st["disc"][have].clamp(min=1e-9), 1).squeeze(-1)
                one = F.one_hot(idx, C.N).float()
                st["disc"][have] -= one
                st["draw"][have] += one
        m = act & (opk == o["move_exhaust_to_hand"])
        if m.any():
            have = m & (st["exh"].sum(-1) > 0.5) & (st["hand"].sum(-1) < C.HAND_MAX)
            if have.any():
                idx = torch.multinomial(st["exh"][have].clamp(min=1e-9), 1).squeeze(-1)
                one = F.one_hot(idx, C.N).float()
                st["exh"][have] -= one
                st["hand"][have] += one
        m = act & (opk == o["play_top_card"]) & ~torch.tensor(nested, device=DEVICE)
        if m.any() and not nested:
            for _ in range(int(amt[m].max().item())):
                have = m & (st["draw"].sum(-1) + st["disc"].sum(-1) > 0.5)
                if not have.any():
                    break
                empty = st["draw"].sum(-1) < 0.5
                resh = have & empty
                st["draw"][resh] += st["disc"][resh]; st["disc"][resh] = 0.0
                idx = torch.multinomial(st["draw"][have].clamp(min=1e-9), 1).squeeze(-1)
                rows2 = torch.zeros(B, dtype=torch.long, device=DEVICE)
                rows2[have] = idx
                one = F.one_hot(rows2, C.N).float() * have.float().unsqueeze(-1)
                st["draw"] -= one
                am = alive_mask(st)
                r = torch.rand(B, E_MAX, device=DEVICE).masked_fill(~am, -1.0)
                tgt2 = r.argmax(-1)
                ex2 = resolve_player_card(C, st, rows2, tgt2, have, nested=True)
                st["exh"] += one                        # Havoc always exhausts it
        m = act & (opk == o["copy_in_hand"])
        if m.any():
            eligible = st["hand"] * ((C.CTYPE == CT_ATTACK) | (C.CTYPE == CT_POWER)).float()
            have = m & (eligible.sum(-1) > 0.5)
            if have.any():
                idx = torch.multinomial(eligible[have].clamp(min=1e-9), 1).squeeze(-1)
                st["hand"][have] += F.one_hot(idx, C.N).float() * amt[have].unsqueeze(-1)
        m = act & (opk == o["upgrade_in_hand"])
        if m.any():
            base_in_hand = st["hand"][:, :C.M]
            do_all = m & (amt < 0)
            up = base_in_hand * do_all.float().unsqueeze(-1)
            one_m = m & (amt > 0) & (base_in_hand.sum(-1) > 0.5)
            if one_m.any():
                idx = torch.multinomial(base_in_hand[one_m].clamp(min=1e-9), 1).squeeze(-1)
                up[one_m] += F.one_hot(idx, C.M).float()
            st["hand"][:, :C.M] -= up
            st["hand"][:, C.M:2 * C.M] += up
        m = act & (opk == o["rampage_grow"])
        st["rampage"] += amt * m.float()
        m = act & (opk == o["multiply_block"])
        st["pblock"] *= torch.where(m, amt.clamp(min=2.0), torch.ones_like(amt))
        m = act & (opk == o["multiply_power"])
        if m.any():
            pid = prog[:, k, 3].long()
            for p_ in pid[m].unique():
                sel = m & (pid == p_)
                st["ppow"][:, p_][sel] *= 2.0
        m = act & (opk == o["generate_random_card"])
        if m.any():
            atk_rows = torch.nonzero(C.ATTACK_ROWS[:C.M]).squeeze(-1)
            idx = atk_rows[torch.randint(0, len(atk_rows), (B,), device=DEVICE)]
            st["hand"] += F.one_hot(idx, C.N).float() * m.float().unsqueeze(-1)
        m = act & (opk == o["set_flag"])
        st["inferno"] = torch.where(m, torch.ones_like(st["inferno"]), st["inferno"])
        fatal_credit |= (pre_alive & ~alive_mask(st)).any(-1) & act
    return exhaust_flag


def _on_exhaust(C, st, who, count):
    fnp = st["ppow"][:, PW["feel_no_pain"]]
    if (fnp > 0).any():
        _player_gain_block(C, st, fnp * (count if torch.is_tensor(count) else float(count)),
                           who & (fnp > 0), from_card=False)
    de = st["ppow"][:, PW["dark_embrace"]]
    if (de > 0).any():
        n = de * (count if torch.is_tensor(count) else float(count))
        draw_cards(C, st, n.long(), who & (de > 0))


# ------------------------------------------------------------ enemy moves
def resolve_enemy_move(C, st, e, active):
    """Execute slot e's telegraphed move against the player."""
    B = st["B"]
    live = active & (st["etype"][:, e] >= 0) & (st["ehp"][:, e] > 0) & player_alive(st)
    if not live.any():
        return
    st["eblk"][:, e][live] = 0.0                        # block clears when it acts
    mv = st["move"][:, e]
    prog = C.MOVES[mv]
    src_hp = st["ehp"][:, e].clone()
    estr = st["epow"][:, e, PW["strength"]]
    eweak = torch.where(st["epow"][:, e, PW["weak"]] > 0,
                        torch.tensor(0.75, device=DEVICE),
                        torch.tensor(1.0, device=DEVICE))
    src_mask = F.one_hot(torch.full((B,), e, dtype=torch.long, device=DEVICE),
                         E_MAX).bool()
    for k in range(MAX_FX):
        opk = prog[:, k, 0].long()
        # NOTE: no etype recheck here -- a move that kills its own source
        # (slime split: die_without_rewards -> spawn) must still finish.
        act = live & (opk != OP["pad"]) & player_alive(st)
        if not act.any():
            continue
        amt = prog[:, k, 1].clone()
        amt = torch.where(amt == AMT_DIVIDER,
                          torch.floor(st["php"].clamp(min=0) / 12) + 1, amt)
        amt = torch.where(amt == AMT_SOURCE_HP, src_hp, amt)
        amt = torch.where(amt == AMT_CAPTURED, st["captured"], amt)
        amt = torch.where(prog[:, k, 1] == AMT_SPAWN,
                          st["spawn_amt"][:, e] + prog[:, k, 7], amt)
        hits = prog[:, k, 2].clamp(min=0)
        tcode = prog[:, k, 4].long()
        o = OP
        m = act & (opk == o["damage"])
        if m.any():
            per_hit = (amt + estr * prog[:, k, 3]) * eweak
            for h in range(HIT_CAP):
                hm = m & (hits > h)
                if not hm.any():
                    break
                _hit_player(C, st, per_hit * hm.float(),
                            source_mask=src_mask & hm.unsqueeze(-1))
        m = act & (opk == o["gain_block"])
        if m.any():
            to_self = m & (tcode == TGT_SELF)
            st["eblk"][:, e] += amt * to_self.float()
            to_ally = m & (tcode == TGT_ALLY)
            if to_ally.any():
                am = alive_mask(st)
                others = am.clone(); others[:, e] = False
                r = torch.rand(B, E_MAX, device=DEVICE).masked_fill(~others, -1.0)
                has_other = others.any(-1)
                pick = F.one_hot(r.argmax(-1), E_MAX).float()
                pick = torch.where(has_other.unsqueeze(-1), pick, src_mask.float())
                st["eblk"] += pick * (amt * to_ally.float()).unsqueeze(-1)
        m = act & (opk == o["apply_power"])
        if m.any():
            pid = prog[:, k, 3].long()
            to_pl = m & (tcode == TGT_PLAYER)
            for p_ in pid[to_pl].unique():
                sel = to_pl & (pid == p_)
                st["ppow"][:, p_][sel] += amt[sel]
            to_self = m & (tcode == TGT_SELF)
            for p_ in pid[to_self].unique():
                sel = to_self & (pid == p_)
                st["epow"][:, e, p_][sel] += amt[sel]
        m = act & (opk == o["create_card"])
        if m.any():
            crow = prog[:, k, 3].long()
            dynb = crow == CARD_DYNAMIC_BURN
            crow = torch.where(dynb & (st["inferno"] > 0),
                               torch.full_like(crow, C.BURN_PLUS_ROW),
                               torch.where(dynb, torch.full_like(crow, C.BURN_ROW), crow))
            onehot = F.one_hot(crow.clamp(min=0), C.N).float() * (amt * m.float()).unsqueeze(-1)
            dest = prog[:, k, 5].long()
            st["draw"] += onehot * (dest == DEST_DRAW).float().unsqueeze(-1)
            st["disc"] += onehot * (dest != DEST_DRAW).float().unsqueeze(-1)
        m = act & (opk == o["heal"])
        st["ehp"][:, e] = torch.minimum(st["ehp"][:, e] + amt * m.float(),
                                        st["emax"][:, e])
        m = act & (opk == o["steal_gold"])
        st["stolen"][:, e] += amt * m.float()
        m = act & (opk == o["capture"])
        st["captured"] = torch.where(m, st["ehp"][:, e], st["captured"])
        m = act & (opk == o["die_no_rewards"])
        if m.any():
            st["etype"][:, e][m] = -1
            st["ehp"][:, e][m] = 0.0
        m = act & (opk == o["escape"])
        if m.any():
            st["gold_lost"] += st["stolen"][:, e] * m.float()
            st["stolen"][:, e][m] = 0.0
            st["etype"][:, e][m] = -1
        m = act & (opk == o["spawn"])
        if m.any():
            kid = prog[:, k, 3].long()
            n_spawn = amt.long()
            hp_over = torch.where(st["captured"] > 0, st["captured"], src_hp)
            use_over = prog[:, k, 5] > 0
            for _ in range(int(n_spawn[m].max().item())):
                need = m & (n_spawn > 0)
                if not need.any():
                    break
                free = (st["etype"] < 0)
                slot = free.float().argmax(-1)
                ok = need & free.any(-1)
                rows = torch.nonzero(ok).squeeze(-1)
                if len(rows) > 0:
                    si = slot[rows]
                    for s_ in si.unique():
                        rr = rows[si == s_]
                        init_slot(C, st, rr, int(s_), kid[rr])
                        ov = use_over[rr]
                        hp = torch.where(ov, torch.floor(hp_over[rr]),
                                         st["ehp"][rr, s_])
                        st["ehp"][rr, s_] = hp
                        st["emax"][rr, s_] = torch.maximum(hp, st["emax"][rr, s_] * 0 + hp)
                        st["hp_budget"][rr] += hp
                n_spawn = n_spawn - need.long()
        m = act & (opk == o["change_state"])   # handled by AI registers
        _ = m
    # end-of-move powers: Ritual (cultist)
    still = live & (st["etype"][:, e] >= 0) & (st["ehp"][:, e] > 0)
    rit = st["epow"][:, e, PW["ritual"]]
    st["epow"][:, e, PW["strength"]] += rit * still.float()
    met = st["epow"][:, e, PW["metallicize"]]
    st["eblk"][:, e] += met * still.float()


# ------------------------------------------------------------ turn phases
def start_player_turn(C, st, first_turn):
    B = st["B"]
    on = player_alive(st) & alive_mask(st).any(-1)
    st["ppow"][:, PW["flame_barrier"]] = 0.0
    barr = st["ppow"][:, PW["barricade"]] > 0
    st["pblock"] = torch.where(barr, st["pblock"], torch.zeros_like(st["pblock"]))
    st["energy"] = torch.full((B,), C.ENERGY, device=DEVICE) \
        + st["ppow"][:, PW["berserk"]]
    st["ppow"][:, PW["strength"]] += st["ppow"][:, PW["demon_form"]] * on.float()
    brut = st["ppow"][:, PW["brutality"]]
    if (brut > 0).any():
        _lose_hp(C, st, brut, on & (brut > 0))
        draw_cards(C, st, brut.long(), on & (brut > 0))
    if first_turn:
        inn = st["draw"] * C.INNATE.unsqueeze(0)
        moved = inn.sum(-1)
        st["draw"] -= inn
        st["hand"] += inn
        n = (C.DRAW - moved).clamp(min=0).long()
        draw_cards(C, st, n, on)
    else:
        draw_cards(C, st, torch.full((B,), C.DRAW, dtype=torch.long, device=DEVICE), on)


def end_player_turn(C, st):
    B = st["B"]
    on = player_alive(st)
    # burn end-of-turn damage (amount x copies in hand)
    for row, amt in ((C.BURN_ROW, 2.0), (C.BURN_PLUS_ROW, 4.0)):
        n = st["hand"][:, row]
        if (n > 0).any():
            _hit_player(C, st, amt * n * on.float())
    # ethereal cards in hand exhaust
    eth = st["hand"] * C.ETHEREAL.unsqueeze(0)
    n_eth = eth.sum(-1)
    st["hand"] -= eth
    st["exh"] += eth
    _on_exhaust(C, st, on & (n_eth > 0), n_eth)
    # discard the hand
    st["disc"] += st["hand"]
    st["hand"] = torch.zeros_like(st["hand"])
    met = st["ppow"][:, PW["metallicize"]]
    if (met > 0).any():
        _player_gain_block(C, st, met, on & (met > 0), from_card=False)
    comb = st["ppow"][:, PW["combust"]]
    if (comb > 0).any():
        who = on & (comb > 0)
        _lose_hp(C, st, torch.ones_like(comb), who)
        _hit_enemies(C, st, comb * who.float(), alive_mask(st) & who.unsqueeze(-1),
                     source_player=True, is_attack=False)
    for pw_ in ("rage", "no_draw", "entangled"):
        st["ppow"][:, PW[pw_]] = 0.0


def end_of_round(C, st):
    for pw_ in ("vulnerable", "weak", "frail"):
        st["ppow"][:, PW[pw_]] = (st["ppow"][:, PW[pw_]] - 1).clamp(min=0)
        st["epow"][..., PW[pw_]] = (st["epow"][..., PW[pw_]] - 1).clamp(min=0)


# ------------------------------------------------------------ player turn
def legal_actions(C, st):
    """(B, N*E_MAX + 1) bool mask. Untargeted cards use slot 0 only."""
    B = st["B"]
    in_hand = st["hand"] >= 1.0
    cost = C.COST.unsqueeze(0).expand(B, -1).clone()
    corrupt = st["ppow"][:, PW["corruption"]] > 0
    is_skill = (C.CTYPE == CT_SKILL).unsqueeze(0)
    cost = torch.where(corrupt.unsqueeze(-1) & is_skill, torch.zeros_like(cost), cost)
    is_x = C.COST.unsqueeze(0) == -1.0
    afford = (cost <= st["energy"].unsqueeze(-1) + 1e-6) | is_x
    ent = st["ppow"][:, PW["entangled"]] > 0
    not_ent = ~(ent.unsqueeze(-1) & (C.CTYPE == CT_ATTACK).unsqueeze(0))
    hand_all_atk = ((st["hand"] * (~C.ATTACK_ROWS).float()).sum(-1) < 0.5)
    clash_ok = ~(C.CLASH.unsqueeze(0) > 0) | hand_all_atk.unsqueeze(-1)
    ok = in_hand & afford & not_ent & clash_ok & (C.UNPLAYABLE.unsqueeze(0) < 0.5)
    am = alive_mask(st)
    targ = C.TARGETED.unsqueeze(0) > 0.5
    per_slot = ok.unsqueeze(-1) & torch.where(
        targ.unsqueeze(-1), am.unsqueeze(1),
        F.one_hot(torch.zeros(1, dtype=torch.long, device=DEVICE), E_MAX)
        .bool().unsqueeze(0))
    mask = torch.cat([per_slot.reshape(B, -1),
                      torch.ones(B, 1, dtype=torch.bool, device=DEVICE)], -1)
    return mask


def player_turn(C, st, policy=None, collect_logp=False):
    B = st["B"]
    logp = torch.zeros(B, device=DEVICE)
    entropy = torch.zeros(B, device=DEVICE)
    done = ~(player_alive(st) & alive_mask(st).any(-1))
    for _ in range(PLAY_CAP):
        if done.all():
            break
        mask = legal_actions(C, st)
        mask[done] = False
        mask[:, -1] = True
        if policy is None:
            logits = torch.zeros(B, C.N * E_MAX + 1, device=DEVICE)
            logits[:, -1] = -0.5
        else:
            logits = policy(C, st, mask)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        if collect_logp:
            logp = logp + dist.log_prob(a) * (~done).float()
            entropy = entropy + dist.entropy() * (~done).float()
        end = (a == C.N * E_MAX) | done
        done = done | end
        playing = ~end
        if not playing.any():
            continue
        row = (a // E_MAX).clamp(max=C.N - 1)
        tgt = (a % E_MAX)
        # pay cost
        cost = C.COST[row]
        corrupt = (st["ppow"][:, PW["corruption"]] > 0) & (C.CTYPE[row] == CT_SKILL)
        eff = torch.where(corrupt, torch.zeros_like(cost), cost)
        is_x = cost == -1.0
        st["x_spent"] = torch.where(is_x, st["energy"], torch.zeros_like(cost)) \
            * playing.float()
        pay = torch.where(is_x, st["energy"], eff.clamp(min=0)) * playing.float()
        st["energy"] = st["energy"] - pay
        one = F.one_hot(row, C.N).float() * playing.float().unsqueeze(-1)
        st["hand"] -= one
        # on-play triggers
        is_atk = (C.CTYPE[row] == CT_ATTACK) & playing
        is_skl = (C.CTYPE[row] == CT_SKILL) & playing
        rage = st["ppow"][:, PW["rage"]]
        if (rage > 0).any():
            _player_gain_block(C, st, rage, is_atk & (rage > 0), from_card=False)
        enr = st["epow"][..., PW["enrage"]]
        st["epow"][..., PW["strength"]] += enr * (is_skl.unsqueeze(-1)
                                                  & alive_mask(st)).float()
        sh = (st["epow"][..., PW["sharp_hide"]] * alive_mask(st).float()).sum(-1)
        if (sh > 0).any():
            _hit_player(C, st, sh * is_atk.float())
        reps = 1 + (is_atk & (st["ppow"][:, PW["double_tap"]] > 0)).long()
        st["ppow"][:, PW["double_tap"]] = (st["ppow"][:, PW["double_tap"]]
                                           - (reps > 1).float()).clamp(min=0)
        exf = resolve_player_card(C, st, row, tgt, playing)
        again = playing & (reps > 1)
        if again.any():
            exf = exf | resolve_player_card(C, st, row, tgt, again)
        # route the card
        is_pow = (C.CTYPE[row] == CT_POWER) & playing
        to_exh = (((C.EXHAUST[row] > 0.5) | exf | corrupt) & playing & ~is_pow)
        st["exh"] += one * to_exh.float().unsqueeze(-1)
        st["disc"] += one * (playing & ~to_exh & ~is_pow).float().unsqueeze(-1)
        _on_exhaust(C, st, to_exh, 1.0)
        done = done | ~(player_alive(st) & alive_mask(st).any(-1))
    return logp, entropy


# ------------------------------------------------------------ full combat
def combat(C, deck_counts, php, enc_ids, policy=None, collect_logp=False):
    B = deck_counts.shape[0]
    st = new_combat(C, deck_counts, php, enc_ids)
    st["x_spent"] = torch.zeros(B, device=DEVICE)
    # guardian shift counter init
    for ti, spec in enumerate(C.AI):
        if spec.special == "guardian":
            g = st["etype"] == ti
            st["regs"][..., 4][g] = spec.shift0
    logp = torch.zeros(B, device=DEVICE)
    entropy = torch.zeros(B, device=DEVICE)
    start_hp = php.clone()
    won = torch.zeros(B, device=DEVICE)
    for t in range(TURN_CAP):
        ongoing = player_alive(st) & alive_mask(st).any(-1)
        if not ongoing.any():
            break
        st["turn"] = t
        st["move"] = ai_mod.choose_moves(C, st["etype"], alive_mask(st).float(),
                                         st["ehp"], st["emax"], st["eblk"],
                                         st["regs"], t)
        start_player_turn(C, st, first_turn=(t == 0))
        lp, en = player_turn(C, st, policy, collect_logp)
        logp = logp + lp
        entropy = entropy + en
        end_player_turn(C, st)
        for e in range(E_MAX):
            resolve_enemy_move(C, st, e, ongoing)
        end_of_round(C, st)
        newly_won = ongoing & ~alive_mask(st).any(-1) & player_alive(st)
        if newly_won.any():
            st["php"] = torch.minimum(
                st["php"] + C.BURNING_BLOOD_HEAL * newly_won.float(),
                torch.full_like(st["php"], C.HP_MAX))
            won = won + newly_won.float()
    won = torch.clamp(won + ((~alive_mask(st).any(-1)) & player_alive(st)).float()
                      - won * 0, max=1.0)
    won = ((~alive_mask(st).any(-1)) & player_alive(st)).float()
    end_hp = st["php"].clamp(min=0) * won
    remaining = (st["ehp"].clamp(min=0) * (st["etype"] >= 0).float()).sum(-1)
    frac = (1.0 - remaining / st["hp_budget"].clamp(min=1.0)).clamp(0.0, 1.0)
    return dict(won=won, end_hp=end_hp, delta_hp=end_hp - start_hp, logp=logp,
                entropy=entropy, frac=frac, gold_lost=st["gold_lost"], st=st)
