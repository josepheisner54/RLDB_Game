"""Fast end-to-end smoke: every trainer runs, losses are finite, shapes hold."""
import torch
from deckbuilder import (train_combat, refine_value, train_meta,
                         simulate_runs, MetaAgent)

torch.manual_seed(0)


def test_pipeline_smoke():
    policy, V, hist = train_combat(steps=3, B=32, eval_every=2, verbose=False)
    assert all(torch.isfinite(torch.tensor(h[1:])).all() for h in hist)
    V = refine_value(policy, V, steps=2, B=32, verbose=False)
    meta, _ = train_meta(V, steps=3, B=32, verbose=False)
    for drafter in [None, "random", meta]:
        r = simulate_runs(policy, drafter, B=64)
        assert 0.0 <= r["win"] <= 1.0
