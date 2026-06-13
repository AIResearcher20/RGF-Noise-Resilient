import yaml
import argparse
import torch
from datasets.loader import load_dataset
from experiments.run_main import run_main
from experiments.run_noise import run_noise

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)

    data = load_dataset(config["dataset"]["name"])

    print("🚀 Running Main Experiment")
    run_main(config, data)

    print("🔥 Running Noise Experiments")
    run_noise(config, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    main(args.config)
