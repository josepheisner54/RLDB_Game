# deckbuilder

Hierarchical RL deckbuilder: JSON-driven content, a two-agent architecture
(combat + meta), and a learned value function as the differentiable interface
between them.

## Architecture

| Component | Job | Trained by |
|---|---|---|
| Combat agent (`agents/policies.py`) | Play fights, maximize ΔHP | REINFORCE + V baseline (actor-critic) |
| Value net (`agents/value.py`) | (entering HP, enemy, deck) → (E[ΔHP], P(death)) | Regression on outcomes |
| Meta agent (`agents/meta.py`) | Draft cards, campfire rest-vs-upgrade | Backprop through frozen V |

The game itself is data: `deckbuilder/content/*.json` defines cards (with
upgrades), enemies (with telegraphed intent systems: cycle or weighted-random
with repeat caps), and encounter pools. JSON fields are exactly the feature
vectors agents read — author content, agents adapt.

## Colab workflow

```python
!git clone https://github.com/josepheisner54/RLDB_Game && pip install -e deckbuilder -q
%load_ext autoreload
%autoreload 2
from deckbuilder import *
policy, V, hist = train_combat(steps=4000, B=2048)
```

Edit package files in Colab's editor (or github.dev) mid-session — autoreload
picks them up. Commit from a cell with a fine-grained PAT stored in Colab
Secrets. Checkpoints go to mounted Drive (gitignored); git holds code and
content only.

## Tests — the war stories

`pytest tests/` — the suite encodes every bug this project has caught, so
they cannot silently return:

- **test_hard_mask_blocks_unowned_cards**: soft action masks are exploitable
  by unbounded policy scores; REINFORCE once learned to play cards not in its
  hand (and inflated a headline result doing it)
- **test_deck_quality_matters**: deck-sensitivity is the falsifiable
  prediction the whole two-agent design rests on
- **test_weighted_pattern_respects_max_repeat / test_cycle_pattern_cycles**:
  enemy AI contracts
- **test_campfire_*:** upgrade conserves deck size, rest caps at max HP,
  unowned upgrades masked
- **test_pipeline_smoke**: every trainer runs end-to-end

## Findings log

1. Meta-built runs ≈ 69% wins vs 42% random-built vs 9% never-built (CPU scale)
2. The meta agent drafts the planted OP card, upgrades Bash, and rests
   conditionally on HP — with the *inverted* pattern (rest when healthy,
   fix the deck when desperate): campfire HP proxies deck strength
3. Rest-vs-upgrade is bang-bang in rest value until tuned (~25–27 band);
   difficulty tuning is decision design — a too-hard finale is a pure
   deck-check where HP decisions can't exist
4. Coverage holes recur: any input a decision can move must be covered by
   V's training distribution (HP down to 5, upgraded deck rows, etc.)
5. Stat jitter (domain randomization) pre-trained upgrade piloting before
   upgrades existed
6. REINFORCE beat pathwise/ST gradients for combat play; differentiability
   earns its keep at the designer/meta layer (through V), not the simulator

## Contribution seams (architecture)

Everything is weight-shared pointwise MLPs; no attention anywhere.
- `MLPCombatPolicy`: per-card scorer that never sees the rest of the hand →
  set-transformer over {hand tokens, state token, enemy token}
- `DeckEncoder`: linear on count vector (DeepSets, identity ϕ) → attention
  pooling over (card-embedding, count) pairs; makes V card-pool-agnostic

Both have documented stubs; both slot in without touching training loops.
