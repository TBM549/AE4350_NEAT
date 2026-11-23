"""Rebuild report evidence from completed matched-baseline runs, without training."""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neat
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.utils import activate_batch
from src.models.enviroment import Enviroment
from src.config.env_config import env_config
from src.evaluation.evaluate_sensitivity import model_runs
from src.evaluation.summarize_flights import run as summarize_flights
from src.evaluation.plot.plot_baseline_flights import set_flight_limits, set_distance_limits, flight_legend

SOURCE = ROOT / "neat_models"
OUT = ROOT / "results/baseline"


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))


def load_network(path, config):
    with path.open("rb") as stream:
        genome = pickle.load(stream)
    return genome, neat.nn.FeedForwardNetwork.create(genome, config)


def structural_shares(net):
    # Absolute path-product sums, including node responses, normalized per output.
    paths = {node: np.eye(len(net.input_nodes))[i] for i, node in enumerate(net.input_nodes)}
    for node, _, _, _, response, links in net.node_evals:
        paths[node] = sum((abs(response * weight) * paths[source] for source, weight in links),
                          np.zeros(len(net.input_nodes)))
    scores = np.array([paths[node] for node in net.output_nodes])
    return 100 * np.divide(scores, scores.sum(axis=1, keepdims=True),
                           out=np.zeros_like(scores), where=scores.sum(axis=1, keepdims=True) > 0)


def run():
    source = model_runs()
    assert len(source["baseline"]) == 3
    OUT.mkdir(parents=True, exist_ok=True)
    performance, episodes, validations, curves, weights, provenance, traces = [], [], [], [], [], [], {}
    for run in source["baseline"]:
        seed = run["training_seed"]
        folder = ROOT / "neat_models" / run["model"]
        config_file = next(folder.glob("config_*.ini"))
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_file))
        assert config.stagnation_config.max_stagnation == 15
        assert config.reproduction_config.elitism == 1 and config.reproduction_config.min_species_size == 1
        validation = json.loads((folder / "validation.json").read_text())
        evaluation = json.loads((folder / "evaluation.json").read_text())
        genome, net = load_network(folder / "winner_genome.pkl", config)
        environment = evaluation["environment"]
        # Recount diagnostics with the current attempt radius, preserving saved dynamics.
        environment["ATTEMPT_DISTANCE_MULTIPLIER"] = env_config["ATTEMPT_DISTANCE_MULTIPLIER"]
        environment["EVADER_CONFIG"]["DATASET_PATH"] = str(ROOT / "moth_dataset/moth_eval_dataset.npz")
        env = Enviroment(environment, seed=evaluation["seed"])
        obs, dones = env.reset()
        positions, targets = [env.pursuer.pos.copy()], [env.evader.full_pos[np.arange(env.num_envs), env.evader.current_idx].copy()]
        durations = np.zeros(env.num_envs)
        while not np.all(dones):
            active = ~dones.copy()
            commands = np.zeros((env.num_envs, 3), dtype=np.float32)
            commands[active] = activate_batch(net, obs[active])
            obs, dones = env.step(commands)
            durations[active] = env.timestep * env.dt
            positions.append(env.pursuer.pos.copy())
            targets.append(env.evader.full_pos[np.arange(env.num_envs), env.evader.current_idx].copy())
        actual = [np.mean(env.caught), np.mean(env.catch_times[env.caught]), np.mean(env.calculate_fitness())]
        archived = [evaluation["catch_rate"], evaluation["mean_catch_time"], evaluation["mean_fitness"]]
        np.testing.assert_allclose(actual, archived, rtol=1e-6, atol=1e-6)
        performance.append(dict(training_seed=seed, validation_capture_pct=100*validation["winner"]["catch_rate"],
                                evaluation_capture_pct=100*actual[0], successful_capture_time_s=actual[1],
                                minimum_distance_m=float(np.mean(env.minimum_distance)),
                                failed_minimum_distance_m=float(np.mean(env.minimum_distance[~env.caught])),
                                fitness=actual[2], nodes=len(genome.nodes),
                                enabled_connections=sum(g.enabled for g in genome.connections.values()),
                                winner_generation=validation["winner"]["generation"]))
        attempt_rows = env.attempts.episode_metrics()
        final_distance = np.linalg.norm(targets[-1] - positions[-1], axis=1)
        for i in range(env.num_envs):
            outcome = "success" if env.caught[i] else "failure"
            end_reason = ("capture" if env.caught[i] else
                          "boundary" if final_distance[i] > env.boundary_distance else "time_limit")
            episodes.append(dict(training_seed=seed, episode=i, caught=bool(env.caught[i]),
                                 duration_s=round(float(durations[i]), 8),
                                 minimum_distance_m=float(env.minimum_distance[i]),
                                 outcome=outcome, end_reason=end_reason,
                                 attempt_distance_m=round(env.attempt_distance, 8),
                                 **attempt_rows[i]))
        traces[seed] = (np.array(positions), np.array(targets), durations, env.caught.copy(), env.minimum_distance.copy(), env.dt)
        np.savez_compressed(OUT / f"flights_seed_{seed}.npz",
                            pursuer=traces[seed][0], moth=traces[seed][1],
                            durations=durations, caught=env.caught,
                            minimum_distance=env.minimum_distance, dt=env.dt,
                            catch_distance=env.catch_distance,
                            attempt_distance=env.attempt_distance)
        validations.append((seed, validation["winner"]))
        fitness = np.loadtxt(next(folder.glob("fitness_*.csv")))[:, 0]
        assert len(fitness) == 100
        history = validation["history"]
        running = [max(h["catch_rate"] for h in history if h["generation"] <= g)*100 for g in range(100)]
        curves.append((fitness, running))
        for rank in range(1, 6):
            _, top_net = load_network(folder / f"top_genome_{rank}.pkl", config)
            shares = structural_shares(top_net)
            groups = [shares[:, :4].sum(axis=1).mean(), shares[:, 4:8].sum(axis=1).mean(), shares[:, 8:].sum(axis=1).mean()]
            axes = [shares[axis, [1+axis, 5+axis, 8+axis]].sum() for axis in range(3)]
            weights.append(dict(training_seed=seed, rank=rank, position_pct=groups[0], velocity_pct=groups[1],
                                acceleration_pct=groups[2], north_same_axis_pct=axes[0],
                                east_same_axis_pct=axes[1], down_same_axis_pct=axes[2]))
        hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
                  [config_file, folder / "winner_genome.pkl", folder / "validation.json", folder / "evaluation.json"]}
        provenance.append(dict(model=folder.name, training_seed=seed, sha256=hashes))
        print(f"Seed {seed}: archived evaluation reproduced", flush=True)

    summary = {key: mean_sd([r[key] for r in performance]) for key in performance[0] if key != "training_seed"}
    weight_summary = {key: mean_sd([r[key] for r in weights]) for key in weights[0] if key not in ("training_seed", "rank")}
    chosen = max(validations, key=lambda pair: (pair[1]["catch_rate"], -pair[1]["mean_catch_time_with_failures"], pair[1]["mean_fitness"], -pair[0]))[0]
    pos, target, durations, caught, minimum, dt = traces[chosen]
    successes = np.flatnonzero(caught)
    median = np.median(durations[successes])
    selected = [int(min(successes, key=lambda i: (abs(durations[i]-median), i))),
                int(min(np.flatnonzero(~caught), key=lambda i: (minimum[i], i)))]
    trajectory_rows = []
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    for row, (i, title) in enumerate(zip(selected, ["Median-time capture", "Closest failed approach"])):
        end = int(round(durations[i]/dt)) + 1
        p, e = pos[:end, i], target[:end, i]
        for step, (p_xyz, e_xyz) in enumerate(zip(p, e)):
            trajectory_rows.append(dict(training_seed=chosen, episode=i, time_s=step*dt,
                pursuer_north_m=p_xyz[0], pursuer_east_m=p_xyz[1], pursuer_down_m=p_xyz[2],
                moth_north_m=e_xyz[0], moth_east_m=e_xyz[1], moth_down_m=e_xyz[2]))
        ax = fig.add_subplot(2, 2, row*2+1, projection="3d")
        for xyz, label, color in [(e, "Moth", "tab:orange"), (p, "Pursuer", "tab:blue")]:
            ax.plot(*xyz.T, label=label, color=color)
            ax.scatter(*xyz[0], marker="o", color=color)
            ax.scatter(*xyz[-1], marker="x", color=color)
        ax.set(xlabel="North (m)", ylabel="East (m)", zlabel="Down (m)", title=f"{title}; episode {i}")
        set_flight_limits(ax, e, p)
        flight_legend(ax)
        ax = fig.add_subplot(2, 2, row*2+2)
        ax.plot(np.arange(end)*dt, np.linalg.norm(e-p, axis=1))
        ax.axhline(.05, color="black", linestyle="--", label="Capture threshold")
        ax.set(xlabel="Time (s)", ylabel="Distance (m)", title=f"Minimum {minimum[i]:.3f} m")
        set_distance_limits(ax, np.arange(end)*dt, np.linalg.norm(e-p, axis=1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Matched baseline; training seed {chosen}")
    fig.savefig(OUT / "trajectories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    learning_rows = []
    for column, (ax, label) in enumerate(zip(axes, ["Best training fitness", "Running-best validation capture (%)"])):
        values = np.array([c[column] for c in curves])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.plot(np.arange(100), mean)
        ax.fill_between(np.arange(100), mean-sd, mean+sd, alpha=.2)
        ax.set(xlabel="Generation", ylabel=label)
        ax.grid(alpha=.25)
        for g in range(100):
            learning_rows.append(dict(metric=label, generation=g, mean=mean[g], sd=sd[g]))
    fig.savefig(OUT / "learning.png", dpi=180)
    plt.close(fig)
    write_csv("trajectories.csv", trajectory_rows)
    write_csv("performance.csv", performance)
    write_csv("episodes.csv", episodes)
    summarize_flights()
    write_csv("weight_shares.csv", weights)
    write_csv("learning.csv", learning_rows)
    (OUT / "summary.json").write_text(json.dumps(dict(performance=summary, weights=weight_summary,
                                                     representative_seed=chosen), indent=2))
    print("Saved baseline metrics, learning curves, weights and flight data.", flush=True)


if __name__ == "__main__":
    run()
