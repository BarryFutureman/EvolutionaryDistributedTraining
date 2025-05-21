import os
import json
import shutil


def build_evolution_data(base_dir):
    evolution_dict = {}
    for folder in os.listdir(base_dir):
        if folder.startswith(your_ip_here):
            dh20_path = os.path.join(base_dir, folder).replace("\\", "/")
            gen_folders = [g for g in os.listdir(dh20_path) if g.startswith("Gen") and not g.endswith("20")]
            print(gen_folders)
            for g in gen_folders:
                gen_num = int(g[3:])
                genome_path = os.path.join(dh20_path, g, "genome.json")
                print(genome_path)
                with open(genome_path, "r") as genome_file:
                    genome_data = json.load(genome_file)

                model_name = genome_data.get("model_path", "")
                dna = genome_data.get("dna", {})
                p1_name = genome_data.get("p1", {}).get("model_path", "")
                p2_name = genome_data.get("p2", {}).get("model_path", "")

                mutated = genome_data.get("dna_mutated", "")
                print(p1_name, p2_name, "<<")
                evolution_dict.setdefault(f"{gen_num}", {})[model_name] = {
                    "dna": dna,
                    "p1": p1_name,
                    "p2": p2_name,
                    "mutated": mutated
                }

    with open(os.path.join(base_dir, "evolution.json"), "w") as f:
        json.dump(evolution_dict, f, indent=4)
    return evolution_dict


if __name__ == "__main__":
    build_evolution_data("./")