from pathlib import Path
import pickle
import sys
import argparse
import json
from copy import deepcopy

import neat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluation.plot.one_flight
from src.config.env_config import env_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay a saved model with its own configuration.")
    parser.add_argument("model", nargs="?", type=Path, help="Model directory (defaults to newest complete model)")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    folder = args.model
    if folder is None:
        models = [p for p in (PROJECT_ROOT / "neat_models").glob("model_*")
                  if p.is_dir() and (p / "winner_genome.pkl").exists()
                  and list(p.glob("config_*.ini")) and list(p.glob("settings_*.json"))]
        if not models:
            parser.error("No complete saved models found; train a controller first.")
        folder = max(models, key=lambda p: (p.stat().st_mtime_ns, p.name))
    winner_path = folder / "winner_genome.pkl"
    config_path = next(folder.glob("config_*.ini"))
    settings = json.loads(next(folder.glob("settings_*.json")).read_text())
    environment = deepcopy(settings.get("environment", env_config))
    # Saved absolute paths may refer to another checkout or operating system.
    dataset_name = str(environment["EVADER_CONFIG"]["DATASET_PATH"]).replace("\\", "/").split("/")[-1]
    local_dataset = PROJECT_ROOT / "moth_dataset" / dataset_name
    if local_dataset.exists():
        environment["EVADER_CONFIG"]["DATASET_PATH"] = str(local_dataset)

    with winner_path.open("rb") as winner_file:
        winner = pickle.load(winner_file)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    return src.evaluation.plot.one_flight.plot_one_flight(
        winner,
        config,
        env_config=environment,
        max_steps=args.max_steps,
        seed=args.seed,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
