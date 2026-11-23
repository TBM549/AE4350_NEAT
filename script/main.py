from pathlib import Path
import pickle
import sys
from datetime import datetime
import random
import json
import argparse

import neat as neat
import numpy as np

from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.enviroment
import src.evaluation.plot.one_flight
from src.evaluation.utils import evaluate_controller
from src.attempts import summarize_attempts
from src.validation import ValidationReporter, split_training_recordings

from src.config.env_config import env_config

def eval_genomes(genomes, config, env):
    scenarios = env.sample_scenarios(resample_recordings=True)
    for genome_id, genome in genomes:
        genome.fitness = evaluate_controller(genome, config, env, scenarios)


def train(seed=2, generations=100, population_size=None, config_path=PROJECT_ROOT / "config", plot_one_flight=False,
          validation_seed=142, validation_episodes=100, validation_candidates=5,
          validation_fraction=0.2, split_seed=2026, model_name=None, environment_config=None):
    if generations < 1 or validation_episodes < 1 or validation_candidates < 1:
        raise ValueError("generations, validation_episodes and validation_candidates must be positive")
    random.seed(seed)
    np.random.seed(seed)

    # Split only the training dataset. The final test dataset is never used
    # for candidate selection, and the split stays the same across training seeds.
    training_config = deepcopy(env_config if environment_config is None else environment_config)
    dataset_path = Path(training_config["EVADER_CONFIG"]["DATASET_PATH"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    partitions = split_training_recordings(dataset_path, split_seed, validation_fraction)
    training_config["EVADER_CONFIG"]["DATASET_PATH"] = str(dataset_path)
    training_config["EVADER_CONFIG"]["TRAJECTORY_INDICES"] = partitions["training_indices"]

    # Initialize environment and starting points
    env = src.models.enviroment.Enviroment(
        config=training_config,
        seed=seed,
    )

    # Load settings from the chosen file
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    # Override only the parameter selected for this experiment
    if population_size is not None:
        neat_config.pop_size = population_size

    # Create population
    p = neat.Population(neat_config)    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Fix windows, initial states and noise on held-out source recordings.
    validation_config = deepcopy(training_config)
    validation_config["EVADER_CONFIG"]["TRAJECTORY_INDICES"] = partitions["validation_indices"]
    validation_config["NUM_ENVS"] = validation_episodes
    validation_env = src.models.enviroment.Enviroment(
        config=validation_config, seed=validation_seed,
    )
    validation_scenarios = validation_env.sample_scenarios()
    validation = ValidationReporter(
        validation_env, validation_scenarios, candidates=validation_candidates,
    )
    p.add_reporter(validation)

    # Run evolution for up to n generations
    p.run(lambda genomes, config: eval_genomes(genomes, config, env),
        generations,
    )
    winner = validation.winner

    # Test the winner
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner genome to a file
    timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")
    model_prefix = f"model_{model_name}" if model_name else "model"
    project_dir = PROJECT_ROOT / "neat_models" / f"{model_prefix}_{timestamp}"
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving winner genome to {project_dir / 'winner_genome.pkl'}")
    with open(project_dir / "winner_genome.pkl", "wb") as winner_file:
        pickle.dump(winner, winner_file, protocol=pickle.HIGHEST_PROTOCOL)
    for rank, (genome, metrics) in enumerate(validation.top_models, start=1):
        with open(project_dir / f"top_genome_{rank}.pkl", "wb") as genome_file:
            pickle.dump(genome, genome_file, protocol=pickle.HIGHEST_PROTOCOL)
    # Save the fitness data to a CSV file
    stats.save_genome_fitness(
        filename=str(
           project_dir / f"fitness_{timestamp}.csv"
        )
    )
    # Save the configuration to a JSON file
    neat_config.save(
        str(project_dir / f"config_{timestamp}.ini")
    )
    settings = {
        "seed": seed,
        "generations": generations,
        "environment": training_config,
        "recording_split": partitions,
        "validation": {
            "seed": validation_seed,
            "episodes": validation_episodes,
            "candidates_per_generation": validation_candidates,
            "environment": validation_config,
            "selection": "catch_rate, then shortest mean catch time with failures at horizon, then fitness",
            "source": "held-out source recordings from training dataset; fixed scenario bank",
        },
    }
    with open(project_dir / f"settings_{timestamp}.json", "w") as settings_file:
        json.dump(settings, settings_file, indent=2)

    with open(project_dir / "validation.json", "w") as validation_file:
        json.dump({"winner": validation.winner_metrics,
                   "top_models": [dict(rank=rank, filename=f"top_genome_{rank}.pkl", **metrics)
                                  for rank, (_, metrics) in enumerate(validation.top_models, start=1)],
                   "history": validation.history},
                  validation_file, indent=2)
    np.savez_compressed(project_dir / "validation_scenarios.npz", **validation_scenarios)

    # evaluate the winner genome
    evaluation_config = deepcopy(training_config)
    evaluation_config["EVADER_CONFIG"].pop("TRAJECTORY_INDICES", None)
    evaluation_config["EVADER_CONFIG"]["DATASET_PATH"] = str(
        PROJECT_ROOT / "moth_dataset" / "moth_eval_dataset.npz"
    )

    evaluation_seed = 42
    evaluation_env = src.models.enviroment.Enviroment(
        config=evaluation_config,
        seed=evaluation_seed,
    )

    evaluation_fitness = evaluate_controller(
        winner, neat_config, evaluation_env
    )
    print(f"Training fitness: {winner.fitness:.3f}")
    print(f"Evaluation fitness: {evaluation_fitness:.3f}")

    catch_rate = float(np.mean(evaluation_env.caught))
    print(f"Evaluation catch rate: {catch_rate:.1%}")

    successful_times = evaluation_env.catch_times[evaluation_env.caught]
    mean_catch_time = (
        float(np.mean(successful_times))
        if successful_times.size > 0
        else None
    )
    print(f"Mean catch time [s]: {mean_catch_time}")
    attempt_metrics = summarize_attempts(evaluation_env.attempts.episode_metrics())
    print(f"Mean attempts: {attempt_metrics['mean_attempts']:.3f}")
    print(f"Mean first attempt time [s]: {attempt_metrics['mean_first_attempt_time_s']}")

    with open(project_dir / "evaluation.json", "w") as evaluation_file:
        json.dump(
            {
                "seed": evaluation_seed,
                "mean_fitness": evaluation_fitness,
                "environment": evaluation_config,
                "catch_rate": catch_rate,
                "mean_catch_time": mean_catch_time,
                **attempt_metrics,
            },
            evaluation_file,
            indent=2,
        )


    # Plot one flight of the winner
    if plot_one_flight:
        src.evaluation.plot.one_flight.plot_one_flight(winner, neat_config, training_config, max_steps=200, seed=42, show=True)

    return winner, {
        "mean_fitness": evaluation_fitness,
        "catch_rate": catch_rate,
        "mean_catch_time": mean_catch_time,
        **attempt_metrics,
        "validation_catch_rate": validation.winner_metrics["catch_rate"],
        "model_path": str(project_dir),
    }

def train_from_model(model_path, generations=None):
    """Repeat the saved training protocol using this checkout's datasets."""
    model_path = Path(model_path)
    settings = json.loads(next(model_path.glob("settings_*.json")).read_text())
    environment = deepcopy(settings["environment"])
    environment["EVADER_CONFIG"]["DATASET_PATH"] = str(PROJECT_ROOT / "moth_dataset/moth_train_dataset.npz")
    validation = settings["validation"]
    partition = settings["recording_split"]
    return train(
        seed=settings["seed"], generations=settings["generations"] if generations is None else generations,
        config_path=next(model_path.glob("config_*.ini")), environment_config=environment,
        validation_seed=validation["seed"], validation_episodes=validation["episodes"],
        validation_candidates=validation["candidates_per_generation"],
        validation_fraction=partition["fraction"], split_seed=partition["seed"],
        model_name="reproduction",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NEAT or repeat a saved training protocol.")
    parser.add_argument("--from-model", type=Path)
    parser.add_argument("--generations", type=int)
    args = parser.parse_args()
    print("Training NEAT controller...")
    if args.from_model:
        winner, evaluation = train_from_model(args.from_model, generations=args.generations)
    else:
        winner, evaluation = train(generations=100 if args.generations is None else args.generations)
    catch_rate = evaluation["catch_rate"]

    print(f"catch rate = {catch_rate:.1%}")
