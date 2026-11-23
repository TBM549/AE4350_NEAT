"""Plot every baseline evaluation episode, grouped by success/failure and seed."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BASELINE = Path(__file__).resolve().parents[3] / "results/baseline"
FLIGHT_LIMITS = dict(xlim=(2.0, 3.0), ylim=(-0.5, 0.5), zlim=(1.5, 0.5))
DISTANCE_LIMITS = dict(xlim=(0, 3), ylim=(0, 2.0))


def expanded_limits(standard, values):
    """Keep standard bounds; extend outwards to whole units when necessary."""
    lower, upper = sorted(standard)
    lower = min(lower, float(np.floor(np.min(values))))
    upper = max(upper, float(np.ceil(np.max(values))))
    return (upper, lower) if standard[0] > standard[1] else (lower, upper)


def set_flight_limits(ax, *paths):
    points = np.concatenate(paths)
    limits = {name: expanded_limits(standard, points[:, axis])
              for axis, (name, standard) in enumerate(FLIGHT_LIMITS.items())}
    ax.set(**limits)
    # Scale the box with the displayed ranges: one metre is equal on every axis.
    ax.set_box_aspect([abs(upper-lower) for lower, upper in limits.values()])


def set_distance_limits(ax, times, distances):
    ax.set(xlim=expanded_limits(DISTANCE_LIMITS["xlim"], times),
           ylim=expanded_limits(DISTANCE_LIMITS["ylim"], distances))


def flight_legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="black", marker="o", linestyle="None", label="Start"),
                Line2D([], [], color="black", marker="x", linestyle="None", label="End")]
    ax.legend(handles=handles, fontsize=8)


def run():
    files = sorted(BASELINE.glob("flights_seed_*.npz"))
    if not files:
        raise FileNotFoundError("Run script/evaluate_baseline.py first.")
    rows = []
    fig = plt.figure(figsize=(8, 4))
    fig.subplots_adjust(wspace=.65)
    trajectory = fig.add_subplot(121, projection="3d")
    distance = fig.add_subplot(122)
    for file in files:
        seed = int(file.stem.split("_")[-1])
        with np.load(file) as data:
            dt = float(data["dt"])
            for episode, caught in enumerate(data["caught"]):
                outcome = "success" if caught else "failed"
                folder = BASELINE / "flights" / outcome / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"episode_{episode:03d}.png"
                end = int(round(data["durations"][episode] / dt)) + 1
                pursuer = data["pursuer"][:end, episode]
                moth = data["moth"][:end, episode]
                trajectory.clear()
                distance.clear()
                for xyz, label, color in ((moth, "Moth", "tab:orange"),
                                           (pursuer, "Pursuer", "tab:blue")):
                    trajectory.plot(*xyz.T, label=label, color=color)
                    trajectory.scatter(*xyz[0], marker="o", color=color)
                    trajectory.scatter(*xyz[-1], marker="x", color=color)
                trajectory.set(xlabel="North (m)", ylabel="East (m)", zlabel="Down (m)")
                set_flight_limits(trajectory, moth, pursuer)
                flight_legend(trajectory)
                distance.plot(np.arange(end)*dt, np.linalg.norm(moth-pursuer, axis=1))
                distance.axhline(float(data["catch_distance"]), color="black",
                                   linestyle="--", label="Capture threshold")
                distance.set(xlabel="Time (s)", ylabel="Distance (m)",
                               title=f"Minimum {data['minimum_distance'][episode]:.3f} m")
                set_distance_limits(distance, np.arange(end)*dt, np.linalg.norm(moth-pursuer, axis=1))
                distance.grid(alpha=.25)
                distance.legend(fontsize=8)
                fig.suptitle(f"Seed {seed} | Episode {episode} | {outcome}")
                fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
                rows.append(dict(training_seed=seed, episode=episode, outcome=outcome,
                                 plot=path.relative_to(BASELINE).as_posix()))
                if (episode+1) % 64 == 0:
                    print(f"Seed {seed}: plotted {episode+1}/{len(data['caught'])} flights", flush=True)
    plt.close(fig)
    with (BASELINE / "flights/index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} flight plots to results/baseline/flights.", flush=True)


if __name__ == "__main__":
    run()
