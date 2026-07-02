"""Enumerations for the compiled STS DSL. Loader compiles JSON -> tensor
micro-programs over these opcodes; the engine executes them batched."""

# --- effect opcodes ---
OPS = ["pad", "damage", "gain_block", "apply_power", "draw", "gain_energy",
       "lose_hp", "heal", "gain_max_hp", "create_card", "exhaust_self",
       "exhaust_random", "exhaust_hand", "move_discard_to_draw",
       "move_exhaust_to_hand", "play_top_card", "copy_in_hand",
       "upgrade_in_hand", "rampage_grow", "multiply_block", "multiply_power",
       "block_per_captured", "capture", "generate_random_card", "set_flag",
       "change_state", "spawn", "steal_gold", "escape", "die_no_rewards",
       "damage_self", "remove_power"]
OP = {name: i for i, name in enumerate(OPS)}

# --- powers ---
POWERS = ["strength", "dexterity", "vulnerable", "weak", "frail",
          "metallicize", "ritual", "demon_form", "combust", "brutality",
          "berserk", "barricade", "juggernaut", "feel_no_pain",
          "dark_embrace", "evolve", "fire_breathing", "rupture", "rage",
          "flame_barrier", "double_tap", "corruption", "no_draw", "curl_up",
          "sharp_hide", "enrage", "spore_cloud", "angry", "thievery",
          "artifact", "asleep", "split", "entangled", "double_damage_next"]
PW = {name: i for i, name in enumerate(POWERS)}
P = len(POWERS)
# loader aliases: strength_down -> strength with negated amount

# --- amount sentinels (negative codes in the amount param) ---
AMT_X = -101.0            # energy spent on an X-cost card
AMT_PLAYER_BLOCK = -102.0 # Body Slam
AMT_RAMPAGE = -103.0      # Rampage's growing damage
AMT_SOURCE_HP = -104.0    # split offspring HP / source.current_hp
AMT_CAPTURED = -105.0     # captured register (split_hp, exhaust count)
AMT_DIVIDER = -106.0      # Hexaghost: floor(player HP / 12) + 1
AMT_PERFECTED = -107.0    # Perfected Strike
AMT_SPAWN = -108.0        # rolled once at spawn (louse bite damage)
CARD_DYNAMIC_BURN = -2    # burn+ after first inferno else burn

# --- target codes ---
TGT_CHOSEN = 0     # the card's chosen enemy / the enemy move's player target
TGT_PLAYER = 1
TGT_ALL_ENEMIES = 2
TGT_RANDOM_EACH_HIT = 3
TGT_SELF = 4       # the acting creature
TGT_ALLY = 5       # chosen living ally, fallback self (shield gremlin)

# --- condition codes (param 6) ---
CND_NONE, CND_TGT_VULN, CND_TGT_INTENT_ATK, CND_ASC17, CND_FATAL = range(5)

# --- create_card destinations (param 5) ---
DEST_DRAW, DEST_DISCARD, DEST_HAND = range(3)

# --- card types ---
CT_ATTACK, CT_SKILL, CT_POWER, CT_STATUS = range(4)

# --- rarities ---
RARITY = {"basic": 0, "common": 1, "uncommon": 2, "rare": 3, "status": 4}

# --- AI kinds ---
AI_WEIGHTED, AI_SEQUENCE, AI_ALTERNATING, AI_SHIELD_GREMLIN, AI_LOOTER, \
    AI_LAGAVULIN, AI_GUARDIAN, AI_CYCLE_INTERRUPT = range(8)

# --- intent codes (for telegraphing) ---
INTENT_ATTACK, INTENT_DEFEND, INTENT_BUFF, INTENT_DEBUFF, INTENT_OTHER = range(5)

MAX_FX = 8      # effect slots per program
NPARAM = 8      # params per slot
E_MAX = 6       # max simultaneous enemies (lots_of_slimes=5, splits<=4)
PLAY_CAP = 10   # max card plays per player turn (compute guard)
HIT_CAP = 8     # max hits resolved per damage effect
TURN_CAP = 20   # combat turn limit; timeout counts as defeat
