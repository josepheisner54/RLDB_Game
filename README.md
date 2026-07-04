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

## STS: full-fidelity Slay the Spire Act 1 (`deckbuilder/sts/`)

The bundle-driven engine: 75 Ironclad cards (150 rows with upgrades + 5
statuses) compiled from an ordered effect DSL (31 ops) into batched tensor
micro-programs; real card economy (draw/hand/discard/exhaust as count
vectors, reshuffle, hand cap 10, ethereal, innate, X-costs); 34 powers with
trigger hooks; 25 Act 1 enemies with their AI state machines (Curl Up,
splits, Lagavulin sleep, Guardian mode shift, Nob enrage, Looter theft,
Sentry artifact); all 20 encounters incl. multi-enemy, elites, and the three
bosses; card rewards with the real drop tables and per-run pity offset; and
an Act 1 run simulator (easy->hard pools, repeat exclusion, campfires).

    from deckbuilder import sts
    C = sts.load(ascension=0)          # ascension resolved at compile time
    policy, V, _ = sts.train_combat(C, steps=2000, B=192)
    V = sts.refine_value(C, policy, V)
    meta, _ = sts.train_meta(C, V)
    sts.simulate_runs(C, policy, meta, B=1000)

Random-play sanity (starter deck, 68 HP): hallway fights winnable, Gremlin
Gang 16%, elites ~0-4%, bosses 0% -- the canonical difficulty gradient.

Documented approximations (each touches <=2 cards/enemies): piles are
count-based, so "top of pile" ops act on a random pile card (Havoc,
Headbutt); "choose a card" effects pick a random eligible card (Armaments,
True Grit, Dual Wield, Warcry); Looter's branch is a coin flip; Searing
Blow has two upgrade levels; Rampage's counter is shared per episode;
batch-exhaust triggers fire amount x count; Feed heals without raising the
run HP cap.

GPU notes: policy logp graphs accumulate over every play of every turn;
memory scales with B x N x E_MAX x hidden. B=128-256 with h=64 fits
comfortably on a T4; drop B before dropping h.

## Findings log (STS era)

7. The frac value head (fraction of enemy HP dealt) restored HP-conditional
   campfire resting -- P(rest) 0.96/0.39/0.01 across low/mid/high HP -- by
   giving V an HP-gradient where P(death) saturates (bosses). Correct
   direction this time, unlike v3a's inversion.
8. Meta drafting is worth +15pp run-win over random on the canon-length act
   (17.6% vs 2.4%); never-draft cannot win at all.
9. SCALING NEGATIVE RESULT: 4x width / 3x depth on the factorized MLP moved
   the boss win rate not at all (0.047 / 0.036 / 0.057). Capacity is not the
   binding constraint; information is -- the per-card scorer never sees the
   rest of the hand. Hence:
10. TokenCombatPolicy: set transformer over per-INSTANCE card tokens in all
    four zones (segment embeddings; repeats are separate tokens), enemy
    tokens with type identity + full power vector, a state token, and a
    learned END-TURN token; shared-QK head scattered amax onto the
    (155 x 6) action grid, learned sink for untargeted cards, drop-in via
    the same forward(C, st, mask) contract. Fine-tuning on harvested real
    decks bought ~nothing while sample_decks matched the meta's taste (the
    v2 DAgger lesson, again).
