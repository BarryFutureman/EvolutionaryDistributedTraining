from env.wb_env import *


def noop_reward(
     env: WarehouseBrawl,
) -> float:
    return 0.0


def stay_in_middle_reward(
        env: WarehouseBrawl,
) -> float:
    # Get player object from the environment
    player: Player = env.objects["player"]

    if player.body.position.x > 1 or player.body.position.x < -1:
        return -0.01
    return 0.0


def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    return 0.0


def on_win_with_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    if player.stocks > 0 and opponent.stocks > 0:
        return 0.0
    elif agent == 'player':
        return 1.0
    return 0.0


def on_lose_penalty(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'opponent':
        return -1.0
    return 0.0


def on_draw_penalty(env: WarehouseBrawl, agent: str) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    if player.stocks == opponent.stocks == 3:
        return -1.0
    return 0.0


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 0.0
    return 1.0


def on_get_knockout_penalty(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    return 0.0


def on_knockout_reward_faster_better(env: WarehouseBrawl, agent: str) -> float:
    if agent != 'player':
        return 2.4 * (1 - (env.steps / env.max_timesteps))
    return 0.0


def on_get_knockout_low_stocks_penalty(env: WarehouseBrawl, agent: str) -> float:
    player: Player = env.objects["player"]
    if agent == 'player':
        if player.stocks > 2:
            return 0.0
        else:
            return -2.0
    return 0.0


def damage_interaction_reward_faster_better(
    env: WarehouseBrawl,
) -> float:
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if damage_dealt > 0:
        return (damage_dealt / 140) * (1 - (env.steps / env.max_timesteps))
    else:
        return 0


def damage_interaction_reward_balanced(
    env: WarehouseBrawl,
) -> float:
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    return (damage_dealt - damage_taken) / 140


def on_ground_attack_penalty(
        env: WarehouseBrawl,
) -> float:
    # Get player object from the environment
    player: Player = env.objects["player"]

    if isinstance(player.state, AttackState) and player.is_on_floor():
        return -0.004 * env.dt
    return 0.0


def attack_in_air_reward(
        env: WarehouseBrawl,
) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    damage_dealt = opponent.damage_taken_this_frame

    if not player.is_on_floor():
        if damage_dealt > 0:
            return 0.4 + damage_dealt / 140
    return 0.0


def damage_in_air_opponent_reward(
        env: WarehouseBrawl,
) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    damage_dealt = opponent.damage_taken_this_frame

    if not opponent.is_on_floor():
        if damage_dealt > 0:
            return 0.4 + damage_dealt / 140
    return 0.0


def damage_from_above_reward(
        env: WarehouseBrawl,
) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    damage_dealt = opponent.damage_taken_this_frame

    if damage_dealt > 0 and player.body.position.y - opponent.body.position.y > 0.05:
        return 1.0
    return 0.0


def back_stab_reward(
        env: WarehouseBrawl,
) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    damage_dealt = opponent.damage_taken_this_frame

    if damage_dealt > 0 and player.facing == opponent.facing:
        return 1.0
    return 0.0


def floor_is_lava_reward(
        env: WarehouseBrawl,
) -> float:
    player: Player = env.objects["player"]

    if player.is_on_floor():
        return -0.004 * env.dt
    return 0.0


def light_attack_penalty(
        env: WarehouseBrawl,
) -> float:
    player: Player = env.objects["player"]

    if player.input.key_status["j"].held:
        return -0.1
    return 0.0


def upward_launch_reward(env: WarehouseBrawl,
) -> float:
    opponent: Player = env.objects["opponent"]

    if opponent.damage_velocity[1] > 5:
        return 1.0
    return 0.0


all_reward_functions = {
    'noop_reward': RewTerm(func=noop_reward, weight=1.0),
    'upward_launch_reward': RewTerm(func=upward_launch_reward, weight=2.0),
    'damage_interaction_reward_faster_better': RewTerm(func=damage_interaction_reward_faster_better, weight=1.0),
    'damage_interaction_reward_balanced': RewTerm(func=damage_interaction_reward_balanced, weight=1.0),
    'attack_in_air_reward': RewTerm(func=attack_in_air_reward, weight=2.0),
    'damage_in_air_opponent_reward': RewTerm(func=damage_in_air_opponent_reward, weight=2.0),
    'back_stab_reward': RewTerm(func=back_stab_reward, weight=2.0),
    'floor_is_lava_reward': RewTerm(func=floor_is_lava_reward, weight=0.01),
    'on_ground_attack_penalty': RewTerm(func=on_ground_attack_penalty, weight=0.01),
    'light_attack_penalty': RewTerm(func=light_attack_penalty, weight=0.02),
    'damage_from_above_reward': RewTerm(func=damage_from_above_reward, weight=0.01),
}

all_signals = {
    'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
    'on_win_with_knockout_reward': ('win_signal', RewTerm(func=on_win_with_knockout_reward, weight=50)),
    'on_lose_penalty': ('win_signal', RewTerm(func=on_lose_penalty, weight=50)),
    'on_draw_penalty': ('win_signal', RewTerm(func=on_draw_penalty, weight=50)),
    'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=10)),
    'on_knockout_reward_faster_better': ('knockout_signal', RewTerm(func=on_knockout_reward_faster_better, weight=10)),
    'on_get_knockout_penalty': ('knockout_signal', RewTerm(func=on_get_knockout_penalty, weight=10)),
    'on_get_knockout_low_stocks_penalty': ('knockout_signal', RewTerm(func=on_get_knockout_low_stocks_penalty, weight=10))
}


dna_to_reward_map = {
    0: 'noop_reward',
    1: 'upward_launch_reward',
    2: 'damage_interaction_reward_faster_better',
    3: 'damage_interaction_reward_balanced',
    4: 'attack_in_air_reward',
    5: 'damage_in_air_opponent_reward',
    6: 'back_stab_reward',
    7: 'floor_is_lava_reward',
    8: 'on_ground_attack_penalty',
    9: 'light_attack_penalty',
    10: 'damage_from_above_reward',
    
    11: 'on_win_reward',
    12: 'on_win_with_knockout_reward',
    13: 'on_lose_penalty',
    14: 'on_draw_penalty',
    15: 'on_knockout_reward',
    16: 'on_knockout_reward_faster_better',
    17: 'on_get_knockout_penalty',
    18: 'on_get_knockout_low_stocks_penalty'
}


def dna_to_reward_functions(dna: list[int]):
    selected_reward_functions = {}
    selected_signals = {}

    for gene in dna:
        reward_name = dna_to_reward_map.get(gene)
        if reward_name in all_reward_functions:
            selected_reward_functions[reward_name] = all_reward_functions[reward_name]
        elif reward_name in all_signals:
            selected_signals[reward_name] = all_signals[reward_name]
            
    print("Selected Reward Functions:", selected_reward_functions,
          "Selected Signals:", selected_signals)

    selected_signals['force_on_get_knockout_penalty'] = ('knockout_signal', RewTerm(func=on_knockout_reward, weight=10))

    return selected_reward_functions, selected_signals
