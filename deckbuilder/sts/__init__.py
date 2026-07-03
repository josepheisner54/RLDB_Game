"""Full-fidelity Slay the Spire Act 1 (Ironclad) RL environment.

    from deckbuilder import sts
    C = sts.load(ascension=0)
    policy, V, hist = sts.train_combat(C, steps=2000, B=192)
    meta, _ = sts.train_meta(C, V)
    sts.simulate_runs(C, policy, meta, B=1000)
"""
from .loader import load
from .engine import combat, new_combat, legal_actions
from .encounters import spawn_encounter
from .rewards import roll_offers, apply_pick, init_pity
from .runsim import simulate_runs, DEFAULT_FLOORS
from .agents import CombatPolicy, RandomPolicy, ValueNet, MetaAgent, \
    encounter_features
from .training import train_combat, refine_value, train_meta, eval_combat, \
    sample_decks, sample_conditions, finetune_on_runs, eval_by_pool
