from pathlib import Path
import configparser
import sys
from tempfile import TemporaryDirectory

import numpy as np

from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.main import train


PARAMETERS = {
    "compatibility_threshold": ("DefaultSpeciesSet", [0.75, 1.0, 1.25, 1.50, 2.0]),
    "node_add_prob": ("DefaultGenome", [0.025, 0.05, 0.10, 0.15, 0.25]),
    "conn_add_prob": ("DefaultGenome", [0.025, 0.05, 0.10, 0.15, 0.25]),
    "weight_mutate_power": ("DefaultGenome", [0.1, 0.15, 0.20, 0.25, 0.50]),
    "survival_threshold": ("DefaultReproduction", [0.1, 0.15, 0.20, 0.25, 0.50]),
}


def run_sweep(seeds=(10, 20, 30), generations=100, population_size=150):
    base_config = configparser.ConfigParser()
    with open(PROJECT_ROOT / "config") as config_file:
        base_config.read_file(config_file)

    # Change one parameter at a time; run the shared baseline only once per seed.
    experiments = [("baseline", None, None)]
    for parameter, (section, values) in PARAMETERS.items():
        for value in values:
            if value != base_config.getfloat(section, parameter):
                experiments.append((parameter, section, value))

    results = []
    with TemporaryDirectory() as directory:
        config_path = Path(directory) / "config"
        for parameter, section, value in experiments:
            print(f"Running sweep for {parameter}={value} with {section}...")
            config = deepcopy(base_config)
            if section is not None:
                config.set(section, parameter, str(value))
            with open(config_path, "w") as config_file:
                config.write(config_file)

            runs = []
            result = {"parameter": parameter, "value": value, "runs": runs}
            results.append(result)
            for seed in seeds:
                experiment_name = "matched_baseline" if section is None else f"{parameter}_{value}"
                print(f"Training {parameter}={value}, population {population_size}, seed {seed}...")
                winner, evaluation = train(
                    seed=seed,
                    generations=generations,
                    population_size=population_size,
                    config_path=config_path,
                    model_name=f"sensitivity_{experiment_name}_seed_{seed}",
                )
                runs.append({"seed": seed, **evaluation})

            catch_rates = [run["validation_catch_rate"] for run in runs]
            print(f"{parameter}={value}: validation catch rate = {np.mean(catch_rates):.1%}")
            if len(catch_rates) > 1:
                print(f"Standard deviation across training seeds = {np.std(catch_rates, ddof=1):.1%}")

    print(f"Sweep completed. Results: {results}")
    return results


if __name__ == "__main__":
    run_sweep()
