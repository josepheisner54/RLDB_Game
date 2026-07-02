"""Meta agent training: both decision types, by backprop through frozen V."""
import torch
import torch.nn.functional as F
from ..state import S, DEVICE
from ..engine.sampling import fight_conditions, enemy_summary
from ..agents.meta import MetaAgent, apply_draft, apply_campfire


def expected_future_value(V, counts, hp, from_fight, B):
    probs = counts / counts.sum(-1, keepdim=True)
    total = 0.0
    for f in range(from_fight, len(S.FIGHTS)):
        etype, hp0, ats = fight_conditions(B, f)
        dhp, pdeath = V(hp, enemy_summary(etype, hp0, ats), probs)
        total = total + dhp - 60.0 * pdeath
    return total


def train_meta(V, steps=500, B=1024, eval_every=125, verbose=True):
    meta = MetaAgent().to(DEVICE)
    opt = torch.optim.Adam(meta.parameters(), lr=1e-3)
    rest_heal = S.RUN["campfire_rest_heal"]
    hist = []
    for s in range(steps):
        counts = S.STARTER.unsqueeze(0).repeat(B, 1)
        hp = torch.rand(B, device=DEVICE) * 45 + 25
        offers = torch.randint(0, S.M, (B, S.RUN["n_offers"]), device=DEVICE)
        w = F.gumbel_softmax(meta.draft_logits(offers, counts, hp), tau=1.0)
        val_d = expected_future_value(V, apply_draft(w, offers, counts), hp, 1, B)

        counts2 = S.STARTER.unsqueeze(0).repeat(B, 1)
        add = F.one_hot(torch.randint(0, S.M, (B,), device=DEVICE), S.N).float()
        counts2 = counts2 + add * (torch.rand(B, device=DEVICE) < 0.8).float().unsqueeze(-1)
        hp2 = torch.rand(B, device=DEVICE) * 50 + 15
        w2 = F.gumbel_softmax(meta.campfire_logits(counts2, hp2), tau=1.0)
        c2, h2 = apply_campfire(w2, counts2, hp2, rest_heal)
        val_c = expected_future_value(V, c2, h2, 2, B)

        loss = -(val_d.mean() + val_c.mean()) / 140.0
        opt.zero_grad(); loss.backward(); opt.step()
        if s % eval_every == 0 or s == steps - 1:
            hist.append((s, -loss.item() * 140))
            if verbose:
                print(f"[meta] step {s:5d}  predicted value {-loss.item() * 140:+.2f} HP")
    return meta, hist
