import matplotlib.pyplot as plt
import neat
import numpy as np
from copy import deepcopy

from src.models.enviroment import Enviroment
from src.config.env_config import env_config
from pathlib import Path


def plot_one_flight(
    winner,
    neat_config,
    env_config = env_config,
    max_steps: int | None = None,
    seed: int = 0,
    show: bool = True,
):
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    env_config = deepcopy(env_config)
    env_config["NUM_ENVS"] = 1  # Only one environment for plotting
    env_config["MAX_TIMESTEPS"] = max_steps if max_steps is not None else env_config["MAX_TIMESTEPS"]
    if env_config["MAX_TIMESTEPS"] < 1:
        raise ValueError("max_steps must be positive")
    dataset = Path(env_config["EVADER_CONFIG"]["DATASET_PATH"])
    if not dataset.is_absolute():
        env_config["EVADER_CONFIG"]["DATASET_PATH"] = str(Path(__file__).resolve().parents[3] / dataset)

    env = Enviroment(config=env_config, seed=seed)
    observations, _ = env.reset()

    steps = env.max_timesteps
    moth_path = np.empty((steps, 3), dtype=np.float32)
    drone_path = np.empty((steps, 3), dtype=np.float32)
    last_step = -1

    for step in range(steps):
        command = np.asarray(winner_net.activate(observations[0]), dtype=np.float32).reshape(1, 3)
        observations, dones = env.step(command)

        states = env.get_states()
        moth_path[step] = states["evader"]["true_position"][0]
        drone_path[step] = states["pursuer"]["true_position"][0]
        last_step = step
        # commands[step] = command[0]
        print(f"Step {step}:")
        print(f"\tDrone pos: {drone_path[step]}, Moth pos: {moth_path[step]}, ")
        print(f"\tDistance: {np.linalg.norm(drone_path[step] - moth_path[step]):.3f}")
        print(f"\tDrone command: {command[0]}")
        print(f"\tDrone observations: {observations[0]}")

        if dones[0]:
            outcome = "caught moth" if env.caught[0] else "episode ended without capture"
            print(f"{outcome} at step {step}, distance: {np.linalg.norm(states['pursuer']['true_position'][0] - states['evader']['true_position'][0]):.3f}")
            break

    valid_len = last_step + 1
    moth_path = moth_path[:valid_len]
    drone_path = drone_path[:valid_len]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(moth_path[:, 0], moth_path[:, 1], moth_path[:, 2], label="Moth")
    ax.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], label="Drone")
    ax.scatter(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], color="red", s=20)
    ax.scatter(*moth_path[0], color="green", s=40)
    ax.scatter(*drone_path[0], color="red", s=40)
    ax.set_xlabel("North")
    ax.set_ylabel("East")
    ax.set_zlabel("Down")
    ax.legend()

    if show:
        plt.show()

    return fig, ax, moth_path, drone_path

