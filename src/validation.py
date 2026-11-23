"""Select an immutable champion on a fixed bank, independently of NEAT fitness."""
from copy import deepcopy
import re

import neat
import numpy as np

from src.evaluation.utils import evaluate_controller


def split_training_recordings(path, seed=2026, fraction=0.2):
    """Reserve whole source recordings for validation, including all their parts."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    with np.load(path, allow_pickle=True) as dataset:
        count = len(dataset["filtered_pos"])
        names = (np.asarray(dataset["filenames"]).astype(str) if "filenames" in dataset
                 else np.arange(count).astype(str))
    groups = np.array([re.sub(r"_part\d+$", "", name) for name in names])
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two source recordings are needed for training and validation")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_validation = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
    held_out = np.isin(groups, shuffled[:n_validation])
    return {
        "seed": seed,
        "fraction": fraction,
        "training_indices": np.flatnonzero(~held_out).tolist(),
        "validation_indices": np.flatnonzero(held_out).tolist(),
        "training_recordings": sorted(set(groups[~held_out])),
        "validation_recordings": sorted(set(groups[held_out])),
    }


def validation_score(metrics):
    return (metrics["catch_rate"], -metrics["mean_catch_time_with_failures"], metrics["mean_fitness"])


class ValidationReporter(neat.reporting.BaseReporter):
    def __init__(self, env, scenarios, candidates=5):
        if candidates < 1:
            raise ValueError("validation_candidates must be positive")
        self.env = env
        self.scenarios = scenarios
        self.candidates = candidates
        self.generation = 0
        self.winner = None
        self.winner_metrics = None
        self.best_score = None
        self.history = []
        self.top_models = []  
        self.validation_cache = {}

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        candidates = sorted(population.values(), key=lambda g: (-g.fitness, g.key))
        max_gnome_metrics = {}
        max_gnome_score = None
        evaluated = 0
        reused = 0
        for genome in candidates[:self.candidates]:
            cached = genome.key in self.validation_cache
            if cached:
                reused += 1
            else:
                fitness = evaluate_controller(genome, config, self.env, self.scenarios)
                catch_rate = float(np.mean(self.env.caught))
                # Assign failures the full horizon. This metric only breaks
                # equal-catch-rate ties.
                duration = float(np.mean(np.where(
                    self.env.caught, self.env.catch_times,
                    self.env.max_timesteps * self.env.dt,
                )))
                self.validation_cache[genome.key] = {
                    "catch_rate": catch_rate,
                    "mean_catch_time_with_failures": duration,
                    "mean_fitness": fitness,
                }
                evaluated += 1
            metrics = {
                "generation": self.generation,
                "genome_id": genome.key,
                "training_fitness": float(genome.fitness),
                **self.validation_cache[genome.key],
                "cached": cached,
            }
            self.history.append(metrics)
            score = validation_score(metrics)
            if not cached and (len(self.top_models) < 5 or score > validation_score(self.top_models[-1][1])):
                self.top_models.append((deepcopy(genome), metrics.copy()))
                self.top_models.sort(key=lambda model: validation_score(model[1]), reverse=True)
                self.top_models = self.top_models[:5]
            if max_gnome_score is None or score > max_gnome_score:
                max_gnome_score = score
                max_gnome_metrics = metrics.copy()

        self.winner, self.winner_metrics = self.top_models[0]
        self.best_score = validation_score(self.winner_metrics)
        print(f"Validation rollouts: {evaluated} new, {reused} cached")
        print(
            f"Best validation candidate: "
            f"{max_gnome_metrics['mean_fitness']:.3f} mean fitness, "
            f"{max_gnome_metrics['catch_rate']:.1%} catches, "
            f"{max_gnome_metrics['mean_catch_time_with_failures']:.3f}s mean catch time "
            f"(genome {max_gnome_metrics['genome_id']})"
        )

        print(
            f"Validation champion: "
            f"{self.winner_metrics['mean_fitness']:.3f} mean fitness, "
            f"{self.winner_metrics['catch_rate']:.1%} catches "
            f"{self.winner_metrics['mean_catch_time_with_failures']:.3f}s mean catch time "
            f"(generation {self.winner_metrics['generation']}, "
            f"genome {self.winner_metrics['genome_id']})"
        )
