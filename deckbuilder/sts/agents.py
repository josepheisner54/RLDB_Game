"""Agents for the STS engine.

CombatPolicy scores every (card row, target slot) pair plus END TURN.
Factorized additive trunk: f(state) + g(card) + h(target) -> shared MLP head.
This is the documented attention seam: replace with a set transformer over
{hand tokens, enemy tokens, state token} without touching the engine, which
only requires __call__(C, st, mask) -> (B, N*E_MAX + 1) logits.

ValueNet: (hp, deck counts, encounter features) -> (E[dHP], P(death)).
Encounter features are precomputed per encounter id (tier, total HP, slots).

MetaAgent: draft head over reward offers (+skip), campfire head over owned
upgradable cards (+REST). Trained by backprop through a frozen ValueNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import PW, E_MAX, INTENT_ATTACK, TURN_CAP, CT_STATUS
from .engine import alive_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DS, DT = 20, 14


def state_features(C, st):
    B = st["B"]
    pp = st["ppow"]
    hand_status = (st["hand"] * (C.CTYPE == CT_STATUS).float()).sum(-1)
    return torch.stack([
        st["php"] / 80, st["energy"] / 3, st["pblock"] / 20,
        pp[:, PW["strength"]] / 5, pp[:, PW["dexterity"]] / 5,
        pp[:, PW["vulnerable"]].clamp(0, 3) / 3, pp[:, PW["weak"]].clamp(0, 3) / 3,
        pp[:, PW["frail"]].clamp(0, 3) / 3,
        (pp[:, PW["corruption"]] > 0).float(), (pp[:, PW["barricade"]] > 0).float(),
        pp[:, PW["demon_form"]] / 3, pp[:, PW["feel_no_pain"]] / 4,
        pp[:, PW["metallicize"]] / 6, hand_status / 5,
        st["hand"].sum(-1) / 10, st["draw"].sum(-1) / 30,
        st["disc"].sum(-1) / 30, st["exh"].sum(-1) / 20,
        torch.full((B,), st["turn"] / TURN_CAP, device=DEVICE),
        alive_mask(st).float().sum(-1) / E_MAX,
    ], -1)


def target_features(C, st):
    am = alive_mask(st).float()
    mv = st["move"].clamp(min=0)
    prog0 = C.MOVES[mv]                     # (B,E,MAX_FX,NPARAM)
    est_dmg = (prog0[..., 1].clamp(min=0) * prog0[..., 2].clamp(min=1)
               * (prog0[..., 0] == 1).float()).sum(-1)
    intent = F.one_hot(C.MOVE_INTENT[mv], 5).float()
    return torch.cat([
        am.unsqueeze(-1), (st["ehp"] / 60).unsqueeze(-1),
        (st["emax"] / 60).unsqueeze(-1), (st["eblk"] / 20).unsqueeze(-1),
        (st["epow"][..., PW["strength"]] / 6).unsqueeze(-1),
        (st["epow"][..., PW["vulnerable"]] > 0).float().unsqueeze(-1),
        (st["epow"][..., PW["weak"]] > 0).float().unsqueeze(-1),
        (est_dmg / 25).unsqueeze(-1) * am.unsqueeze(-1),
        intent * am.unsqueeze(-1),
        (st["epow"][..., PW["artifact"]] > 0).float().unsqueeze(-1),
    ], -1)


class CombatPolicy(nn.Module):
    def __init__(self, C, h=64, n_layers=1):
        """h: trunk/head width. n_layers: hidden Linear(h,h) layers in the
        shared head -- where state x card x target interaction modeling
        lives. Defaults (h=64, n_layers=1) reproduce the original network,
        so existing checkpoints load unchanged."""
        super().__init__()
        self.fs = nn.Linear(DS, h)
        self.fc = nn.Linear(C.FEAT_DIM, h)
        self.ft = nn.Linear(DT, h)
        layers = [nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers += [nn.Linear(h, 1)]
        self.head = nn.Sequential(*layers)
        self.end = nn.Sequential(nn.Linear(DS, h), nn.ReLU(), nn.Linear(h, 1))
        self.ckpt = True          # gradient-checkpoint the head (exact; ~30% fwd recompute)
        self._ccache = None       # (fc weight version, embedding) for no-grad passes

    def _card_emb(self, C):
        """fc(CARD_FEATS) is static within an optimizer step. For no-grad
        passes, cache it keyed on the weight tensor's in-place version
        counter (optimizer steps bump it), so the cache can never go stale."""
        if torch.is_grad_enabled():
            return self.fc(C.CARD_FEATS)
        v = self.fc.weight._version
        if self._ccache is None or self._ccache[0] != v:
            with torch.no_grad():
                self._ccache = (v, self.fc(C.CARD_FEATS))
        return self._ccache[1]

    def _score(self, C, sfeat, tfeat):
        """Pure function of feature SNAPSHOTS -- safe to recompute at backward
        time even though the live combat state has mutated since the play."""
        B = sfeat.shape[0]
        s = self.fs(sfeat)
        c = self._card_emb(C)
        t = self.ft(tfeat)
        x = (s.view(B, 1, 1, -1) + c.view(1, C.N, 1, -1) + t.view(B, 1, E_MAX, -1))
        scores = self.head(x).squeeze(-1).reshape(B, C.N * E_MAX)
        return torch.cat([scores, self.end(sfeat)], -1)

    def forward(self, C, st, mask):
        sfeat = state_features(C, st)
        tfeat = target_features(C, st)
        if self.ckpt and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda a, b: self._score(C, a, b), sfeat, tfeat,
                              use_reentrant=False)
        return self._score(C, sfeat, tfeat)


class RandomPolicy(nn.Module):
    """Uniform over legal actions; the honest baseline."""
    def forward(self, C, st, mask):
        z = torch.zeros(st["B"], C.N * E_MAX + 1, device=DEVICE)
        z[:, -1] = -0.5
        return z


def encounter_features(C):
    """Static per-encounter features for the value net (n_enc, 6)."""
    ids = list(C.ENCOUNTERS.keys())
    feats, idx = [], {}
    tiers = {"easy": 0, "hard": 0, "elite": 1, "boss": 2}
    tier_of = {}
    for pool, t in (("easy_hallway", 0), ("hard_hallway", 0), ("elite", 1), ("boss", 2)):
        for p in C.POOLS[pool]:
            tier_of[p["encounter"]] = t
    for i, eid in enumerate(ids):
        e = C.ENCOUNTERS[eid]
        if e["kind"] == "fixed":
            hps = [C.E_HP[C.eid[s["enemy"]]].mean() for s in e["enemies"]]
        elif e["kind"] == "generated":
            hps = [torch.stack([C.E_HP[C.eid[n]].mean() for n in s["choose_uniform"]]).mean()
                   for s in e["slots"]]
        elif e["kind"] == "choose_uniform_formation":
            hps = [torch.stack([C.E_HP[C.eid[s["enemy"]]].mean()
                                for s in form]).sum() / len(e["formations"])
                   for form in e["formations"]]
        else:
            hps = [torch.stack([C.E_HP[C.eid[n]].mean() for n in e["bag"]]).mean()
                   ] * e["count"]
        total = torch.stack([torch.as_tensor(h) for h in hps]).sum()
        t = tier_of.get(eid, 0)
        feats.append(torch.tensor([total / 150.0, len(hps) / E_MAX,
                                   float(t == 0), float(t == 1), float(t == 2),
                                   max(float(h) for h in hps) / 150.0]))
        idx[eid] = i
    return torch.stack(feats).to(DEVICE), idx


class DeckEncoder(nn.Module):
    """Linear on normalized counts. SEAM: attention pooling over
    (card embedding, count) pairs makes this card-pool-agnostic."""
    def __init__(self, C, d=48):
        super().__init__()
        self.proj = nn.Linear(C.N, d)

    def forward(self, counts):
        return F.relu(self.proj(counts / counts.sum(-1, keepdim=True).clamp(min=1)))


class ValueNet(nn.Module):
    def __init__(self, C, d=128):
        super().__init__()
        self.enc = DeckEncoder(C)
        self.net = nn.Sequential(nn.Linear(1 + 6 + 48, d), nn.ReLU(),
                                 nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 3))

    def forward(self, hp, enc_feats, deck):
        """-> (E[dHP], P(death), E[frac of enemy HP dealt]).
        The frac head is dense credit precisely where the win signal
        saturates (bosses): it still depends on entering HP, so campfire
        decisions keep a gradient even in mostly-lethal fights."""
        x = torch.cat([hp.unsqueeze(-1) / 80, enc_feats, self.enc(deck)], -1)
        o = self.net(x)
        return o[:, 0] * 80, torch.sigmoid(o[:, 1]), torch.sigmoid(o[:, 2])


class MetaAgent(nn.Module):
    def __init__(self, C, d=96):
        super().__init__()
        self.enc = DeckEncoder(C)
        self.draft = nn.Sequential(nn.Linear(C.FEAT_DIM + 48 + 1, d), nn.ReLU(),
                                   nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.skip = nn.Parameter(torch.tensor([0.0]))
        self.up = nn.Sequential(nn.Linear(2 * C.FEAT_DIM + 48 + 1, d), nn.ReLU(),
                                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.rest = nn.Sequential(nn.Linear(48 + 1, d), nn.ReLU(), nn.Linear(d, 1))

    def _ctx(self, deck, hp):
        return torch.cat([self.enc(deck), hp.unsqueeze(-1) / 80], -1)

    def draft_logits(self, C, offers, deck, hp):
        B = deck.shape[0]
        ctx = self._ctx(deck, hp)
        of = C.CARD_FEATS[offers]                              # (B,3,F)
        x = torch.cat([of, ctx.unsqueeze(1).expand(B, offers.shape[1], -1)], -1)
        return torch.cat([self.draft(x).squeeze(-1),
                          self.skip.unsqueeze(0).expand(B, 1)], -1)

    def campfire_logits(self, C, deck, hp):
        B = deck.shape[0]
        ctx = self._ctx(deck, hp)
        pair = torch.cat([C.CARD_FEATS[:C.M], C.CARD_FEATS[C.M:2 * C.M]], -1)
        x = torch.cat([pair.unsqueeze(0).expand(B, C.M, -1),
                       ctx.unsqueeze(1).expand(B, C.M, -1)], -1)
        up = self.up(x).squeeze(-1)
        up = up.masked_fill(deck[:, :C.M] < 0.5, -1e9)
        return torch.cat([up, self.rest(ctx)], -1)
