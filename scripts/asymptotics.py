from geometric_asymptotics.experiment import run_experiment
from pathlib import Path
import tomli
import argparse
import jax

def main(config_file: str):
    print(f"Current directory: {Path.cwd()}")

    config_path = Path.cwd() / "config" / config_file
    print(f"Reading config file: {config_path}")
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    jax.config.update("jax_enable_x64", True)

    for seed in config["seeds"]:
        print(f"Running seed {seed}")
        run_experiment(seed, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GeometricAsymptoticsExperiments",
        usage="asymptotics.py [dragon.toml/dumbbell.toml/sphere.toml]",
        )
    parser.add_argument("config_file")
    args = parser.parse_args()
    main(args.config_file)