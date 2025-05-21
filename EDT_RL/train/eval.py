from agent import TransformerAgent
from reward_functions import *
import json
import os
import random


def run_eval(agent_paths):
    levels = {path: 0 for path in agent_paths}
    round_number = 1

    eval_dir = "eval"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    def run_match_and_update(p1_path, p2_path):
        p1 = TransformerAgent(p1_path)
        p2 = TransformerAgent(p2_path)
        match_stats = run_match(p1, p2, max_timesteps=90*30, reward_manager=None,
                                # video_path=f"{p1_path.split('/')[-2]}vs{p2_path.split('/')[-2]}.mp4",
                                agent_1_name=f"{p1_path.split('/')[-2]}",
                                agent_2_name=f"{p2_path.split('/')[-2]}")

        if match_stats.player1_result == Result.WIN:
            loser = p2_path
            winner = p1_path
        else:
            loser = p1_path
            winner = p2_path
        levels[loser] = round_number
        return winner

    def run_tournament_round(agents):
        winners = []
        for i in range(0, len(agents), 2):
            if i + 1 < len(agents):
                winner = run_match_and_update(agents[i], agents[i + 1])
                winners.append(winner)
            else:
                winners.append(agents[i])
        return winners

    remaining_agents = agent_paths[:]
    while len(remaining_agents) > 1:
        random.shuffle(remaining_agents)
        remaining_agents = run_tournament_round(remaining_agents)
        round_number += 1

    winner = remaining_agents[0]
    levels[winner] = round_number

    with open(os.path.join(eval_dir, "eval_result.json"), "w") as f:
        json.dump(levels, f, indent=2)

    print(levels)
    draw_tournament_results(agent_paths, levels)


def draw_tournament_results(agent_paths, levels):
    sorted_agents = sorted(agent_paths, key=lambda x: levels[x], reverse=True)
    print("\nTournament Results:")
    for agent in sorted_agents:
        print(f"Level {levels[agent]}: {agent}")


def build_evolution_data(base_dir):
    evolution_dict = {}
    for folder in os.listdir(base_dir):
        if folder.startswith("dh20"):
            dh20_path = os.path.join(base_dir, folder).replace("\\", "/")
            gen_folders = [g for g in os.listdir(dh20_path) if g.startswith("Gen")]
            print(gen_folders)
            for g in gen_folders:
                gen_num = int(g[3:])
                m_folder = [f for f in os.listdir(os.path.join(dh20_path, g)) if
                            os.path.isdir(os.path.join(dh20_path, g))][0]
                if gen_num % 2 == 0:
                    genome_path = os.path.join(dh20_path, g, m_folder, "genome.json")
                    print(genome_path)
                    with open(genome_path, "r") as genome_file:
                        genome_data = json.load(genome_file)

                    model_name = f"SLOT_{dh20_path.split('/')[-1]}"
                    dna = genome_data.get("env", {}).get("reward_dna", {})
                    p1_name = genome_data.get("p1", {}).get("model_name", "")
                    p2_name = genome_data.get("p2", {}).get("model_name", "")

                    mutated = False
                    if genome_data.get("env", {}).get("reward_dna_before_mutation"):
                        mutated = True
                    print(p1_name, p2_name, "<<")
                    evolution_dict.setdefault(f"{gen_num/2}", {})[model_name] = {
                        "dna": dna,
                        "p1": p1_name,
                        "p2": p2_name,
                        "mutated": mutated
                    }
                    print(evolution_dict)
    return evolution_dict


if __name__ == '__main__':
    base_dir = "../"
    eval_dir = "eval"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # evolution_data = build_evolution_data(base_dir)
    # with open(os.path.join(eval_dir, "evolution.json"), "w") as f:
    #     json.dump(evolution_data, f, indent=2)

    agent_paths = []
    for folder in os.listdir(base_dir):
        if folder.startswith("dh20"):
            dh20_path = os.path.join(base_dir, folder).replace("\\", "/")
            gen_folders = [f for f in os.listdir(dh20_path) if f.startswith("Gen")]
            if gen_folders:
                latest_gen_folder = max(gen_folders, key=lambda x: int(x[3:]))
                latest_gen_path = os.path.join(dh20_path, latest_gen_folder).replace("\\", "/")
                model_folders = [f for f in os.listdir(latest_gen_path) if
                                 os.path.isdir(os.path.join(latest_gen_path, f))]
                if model_folders:
                    model_folder = model_folders[0]
                    policy_path = os.path.join(latest_gen_path, model_folder, "Policy").replace("\\", "/")
                    agent_paths.append(policy_path)

    print(agent_paths)

    run_eval(agent_paths)
