"""Full runs: fight -> heal+DRAFT -> fight -> heal+CAMPFIRE -> fight."""
import torch
import torch.nn.functional as F
from .state import S, DEVICE, CFG
from .engine.sampling import fight_conditions
from .engine.combat import combat
from .agents.meta import MetaAgent, apply_draft, apply_campfire


def simulate_runs(policy, drafter, B=2000, seed=7, record=False):
    torch.manual_seed(seed)
    run = S.RUN
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    hp = torch.full((B,), CFG["hp_max"], device=DEVICE)
    alive = torch.ones(B, device=DEVICE)
    rec = dict(draft=torch.zeros(S.N + 1), upgrade=torch.zeros(S.M + 1),
               camp_hp=[], camp_rest=[])
    for fight in range(run["n_fights"]):
        probs = counts / counts.sum(-1, keepdim=True)
        etype, hp0, ats = fight_conditions(B, fight)
        with torch.no_grad():
            out = combat(policy, probs, etype, hp0, ats, hp.clamp(min=1))
        alive = alive * out["won"]
        hp = out["end_hp"] * alive
        if fight == run["n_fights"] - 1:
            break
        hp = (hp + run["post_fight_heal"]).clamp(max=CFG["hp_max"]) * alive
        if fight == 0:
            offers = torch.randint(0, S.M, (B, run["n_offers"]), device=DEVICE)
            if drafter == "random":
                w = F.one_hot(torch.randint(0, run["n_offers"] + 1, (B,), device=DEVICE),
                              run["n_offers"] + 1).float()
                counts = apply_draft(w, offers, counts)
            elif isinstance(drafter, MetaAgent):
                with torch.no_grad():
                    w = F.gumbel_softmax(drafter.draft_logits(offers, counts, hp.clamp(min=1)),
                                         tau=1.0, hard=True)
                counts = apply_draft(w, offers, counts)
                if record:
                    rec["draft"][:S.N] += ((w[:, :-1].unsqueeze(-1) * F.one_hot(offers, S.N).float())
                                           .sum(1).sum(0).cpu())
                    rec["draft"][S.N] += w[:, -1].sum().cpu()
        else:
            if drafter == "random":
                rest = (torch.rand(B, device=DEVICE) < 0.5).float()
                pick = torch.multinomial(counts[:, :S.M].clamp(min=1e-8), 1).squeeze(-1)
                w = F.one_hot(pick, S.M + 1).float() * (1 - rest).unsqueeze(-1)
                w[:, S.M] = rest
                counts, hp = apply_campfire(w, counts, hp.clamp(min=1), run["campfire_rest_heal"])
                hp = hp * alive
            elif isinstance(drafter, MetaAgent):
                with torch.no_grad():
                    w = F.gumbel_softmax(drafter.campfire_logits(counts, hp.clamp(min=1)),
                                         tau=1.0, hard=True)
                if record:
                    rec["camp_hp"].append(hp.clone().cpu())
                    rec["camp_rest"].append(w[:, S.M].clone().cpu())
                    rec["upgrade"] += w.sum(0).cpu()
                counts, hp = apply_campfire(w, counts, hp.clamp(min=1), run["campfire_rest_heal"])
                hp = hp * alive
    out = dict(win=alive.mean().item(), final_hp=(hp * alive).mean().item())
    if record:
        out.update(rec)
    return out
