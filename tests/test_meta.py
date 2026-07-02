import torch
import torch.nn.functional as F
from deckbuilder import S, MetaAgent, apply_campfire, apply_draft

torch.manual_seed(0)


def test_campfire_rest_caps_at_max_hp():
    B = 8
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    w = torch.zeros(B, S.M + 1); w[:, S.M] = 1.0          # pure rest
    _, hp = apply_campfire(w, counts, torch.full((B,), 60.0), rest_heal=27)
    assert (hp <= 70.0 + 1e-5).all()


def test_campfire_upgrade_conserves_deck_size():
    B = 8
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    w = torch.zeros(B, S.M + 1); w[:, S.NAMES.index("Bash")] = 1.0
    new, _ = apply_campfire(w, counts, torch.full((B,), 60.0), rest_heal=27)
    assert torch.allclose(new.sum(-1), counts.sum(-1))
    assert new[0, S.NAMES.index("Bash")] == counts[0, S.NAMES.index("Bash")] - 1
    assert new[0, S.M + S.NAMES.index("Bash")] == 1


def test_campfire_masks_unowned_upgrades():
    meta = MetaAgent()
    B = 4
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    logits = meta.campfire_logits(counts, torch.full((B,), 40.0))
    unowned = [i for i in range(S.M) if counts[0, i] < 1]
    assert (logits[:, unowned] < -1e8).all()


def test_draft_adds_exactly_one_card():
    B = 8
    counts = S.STARTER.unsqueeze(0).repeat(B, 1)
    offers = torch.randint(0, S.M, (B, 3))
    w = F.one_hot(torch.zeros(B, dtype=torch.long), 4).float()   # take offer 0
    new = apply_draft(w, offers, counts)
    assert torch.allclose(new.sum(-1), counts.sum(-1) + 1)
