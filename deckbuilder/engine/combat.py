"""Exact discrete combat vs JSON-defined enemies, vectorized over episodes.

Invariants guarded by tests/test_engine.py -- do not relax:
- HARD action mask (masked_fill -1e9). Soft masks (logits + log(mask+eps))
  are exploitable: policy scores are unbounded, so RL learns scores beyond
  the penalty and plays cards NOT IN ITS HAND. We caught it doing this.
- The current intent is telegraphed in the state; enemy block from an
  intent protects the enemy during the player's NEXT turn.
"""
import torch
import torch.nn.functional as F
from ..state import (S, DEVICE, CFG, COST, DMG, HITS, BLK, VULN, WEAK, STR,
                     I_ATK, I_HITS, I_BLK, I_STR, I_HEAL)


def pick_intents(etype, t, last_idx, rep_ct):
    """'cycle' loops the intent list; 'weighted_random' samples under a
    max_repeat cap (how StS enemy AI actually works)."""
    n = S.N_INT[etype]
    cyc_idx = (torch.as_tensor(t, device=DEVICE) % n).long()
    w = S.WEIGHTS[etype].clone()
    block_rep = (rep_ct >= S.MAX_REP[etype]).float()
    w = w * (1 - block_rep.unsqueeze(-1) * F.one_hot(last_idx, S.K).float())
    rnd_idx = torch.multinomial(w.clamp(min=1e-8), 1).squeeze(-1)
    idx = torch.where(S.IS_CYCLE[etype] > 0, cyc_idx, rnd_idx)
    rep_ct = torch.where(idx == last_idx, rep_ct + 1, torch.ones_like(rep_ct))
    return idx, rep_ct


def combat(policy, deck_probs, etype, hp0, atk_scale, start_hp,
           feats=None, collect_logp=False, cfg=CFG):
    B = deck_probs.shape[0]
    T, plays = cfg["turns"], cfg["plays_per_turn"]
    if feats is None:
        feats = S.FEATS.unsqueeze(0).expand(B, S.N, 7)
    fnorm = feats / 10.0
    cost_c, dmg_c, hits_c = feats[..., COST], feats[..., DMG], feats[..., HITS]
    blk_c, vuln_c, weak_c, str_c = feats[..., BLK], feats[..., VULN], feats[..., WEAK], feats[..., STR]
    is_atk = (dmg_c > 0.5).float()

    php, ehp = start_hp.clone(), hp0.clone()
    eblk = torch.zeros(B, device=DEVICE); estr = torch.zeros(B, device=DEVICE)
    vuln = torch.zeros(B, device=DEVICE); weak = torch.zeros(B, device=DEVICE)
    pstr = torch.zeros(B, device=DEVICE)
    last_idx = torch.zeros(B, dtype=torch.long, device=DEVICE)
    rep_ct = torch.zeros(B, dtype=torch.long, device=DEVICE)
    logp = torch.zeros(B, device=DEVICE)

    for t in range(T):
        active = ((php > 0) & (ehp > 0)).float()
        idx, rep_ct = pick_intents(etype, t, last_idx, rep_ct)
        last_idx = idx
        it = S.INTENTS[etype, idx]
        hand_idx = torch.multinomial(deck_probs.clamp(min=1e-8), cfg["hand_size"], replacement=True)
        hand = F.one_hot(hand_idx, S.N).sum(1).float()
        energy = torch.full((B,), cfg["energy"], device=DEVICE)
        pblock = torch.zeros(B, device=DEVICE)
        for _ in range(plays):
            weak_mult = 1 - 0.25 * weak.clamp(0, 1)
            inc = F.relu((it[:, I_ATK] * atk_scale + estr) * it[:, I_HITS].clamp(min=0) * weak_mult)
            mask = ((hand >= 1) & (cost_c <= energy.unsqueeze(1) + 1e-4)).float()
            state = torch.stack([php / 70, ehp / hp0, hp0 / 100, eblk / 10, estr / 6,
                                 vuln.clamp(0, 3) / 3, weak.clamp(0, 3) / 3,
                                 inc / 30, it[:, I_BLK] / 10, it[:, I_STR] / 4,
                                 energy / 3, pblock / 20, pstr / 6,
                                 torch.full((B,), t / T, device=DEVICE)], -1)
            inp = torch.cat([state.unsqueeze(1).expand(B, S.N, 14), fnorm], -1)
            scores = policy(inp).squeeze(-1)
            logits = torch.cat([scores.masked_fill(mask == 0, -1e9),
                                policy.pass_bias.unsqueeze(0).expand(B, 1)], 1)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            if collect_logp:
                logp = logp + dist.log_prob(a) * active
            pick = F.one_hot(a, S.N + 1).float()
            play = pick[:, :S.N] * active.unsqueeze(1)
            dmg = ((play * (dmg_c + pstr.unsqueeze(1) * is_atk) * hits_c).sum(-1)
                   * (1 + 0.5 * vuln.clamp(0, 1)))
            absorbed = torch.minimum(dmg, eblk)
            eblk = eblk - absorbed
            ehp = ehp - (dmg - absorbed)
            pblock = pblock + (play * blk_c).sum(-1)
            vuln = vuln + (play * vuln_c).sum(-1)
            weak = weak + (play * weak_c).sum(-1)
            pstr = pstr + (play * str_c).sum(-1)
            energy = energy - (play * cost_c).sum(-1)
            hand = hand - play
        e_alive = (ehp > 0).float()
        weak_mult = 1 - 0.25 * weak.clamp(0, 1)
        hit = F.relu((it[:, I_ATK] * atk_scale + estr) * weak_mult - pblock / it[:, I_HITS].clamp(min=1))
        php = php - hit * it[:, I_HITS] * e_alive * (php > 0).float()
        estr = estr + it[:, I_STR] * e_alive
        ehp = torch.minimum(ehp + it[:, I_HEAL] * e_alive, hp0)
        eblk = it[:, I_BLK] * e_alive
        vuln = F.relu(vuln - 1); weak = F.relu(weak - 1)

    won = ((ehp <= 0) & (php > 0)).float()
    end_hp = php.clamp(min=0) * won
    return dict(won=won, end_hp=end_hp, delta_hp=end_hp - start_hp, logp=logp)
