from pathlib import Path

import numpy as np

from src.config.moth_config import moth_config


class VecMothEvader():
    def __init__(
        self,
        dt: float,
        num_envs: int,
        max_timesteps: int = 200,
        config: dict = moth_config,
        randomize_on_reset: bool = False,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.max_timesteps = max_timesteps
        self.num_envs = num_envs
        self.dataset_path = Path(config["DATASET_PATH"])
        self.noise_std = config["NOISE_STD"]
        self.randomize_on_reset = randomize_on_reset
        self.allowed_indices = config.get("TRAJECTORY_INDICES")

        self.full_pos, self.full_vel, self.traj_lengths = self._load_and_select_trajectories()
        self.current_idx = np.zeros(self.num_envs, dtype=int)
        self._set_start_index()

    def _get_random_start_index(self, idx=None) -> np.ndarray:
        if idx is None:
            idx = np.arange(self.num_envs)
        lengths = np.asarray(self.traj_lengths)[idx]
        start_idx = self.rng.integers(0, lengths - self.max_timesteps)
        self.current_idx[idx] = start_idx

    def _set_start_index(self, idx = None) -> int:
        if idx is None:
            idx = np.arange(self.num_envs)
        if self.randomize_on_reset:
            self._get_random_start_index(idx)
        else:
            self.current_idx[idx] = 0


    def _load_and_select_trajectories(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Moth dataset not found: {self.dataset_path}")

        dataset = np.load(self.dataset_path, allow_pickle=True)
        if "times" not in dataset:
            raise ValueError("Moth dataset must contain timestamps to verify playback timing")
        for times in dataset["times"]:
            spacing = np.diff(np.asarray(times, dtype=np.float64))
            if spacing.size == 0 or not np.allclose(spacing, self.dt, rtol=0, atol=1e-8):
                raise ValueError(
                    f"Moth dataset timestep does not match simulation dt={self.dt}. "
                    "Resample the dataset first."
                )
        required_keys = ("filtered_pos", "filtered_vel")
        missing_keys = [key for key in required_keys if key not in dataset]
        if missing_keys:
            raise KeyError(
                f"Moth dataset {self.dataset_path} is missing keys: {missing_keys}"
            )

        positions, velocities = dataset["filtered_pos"], dataset["filtered_vel"]
        indices = (np.arange(len(positions)) if self.allowed_indices is None
                   else np.asarray(self.allowed_indices, dtype=int))
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(positions)):
            raise ValueError("TRAJECTORY_INDICES must select valid recordings")
        self.positions_by_index = {}
        self.velocities_by_index = {}
        for index in indices:
            pos, vel = positions[index], velocities[index]
            if pos.shape != vel.shape:
                raise Warning(
                    f"Position and velocity arrays must have the same shape, "
                    f"but got {pos.shape} and {vel.shape}"
                )
            if pos.shape[0] < self.max_timesteps + 1:
                raise Warning(
                    f"Trajectory length must be at least {self.max_timesteps + 1}, "
                    f"but got {pos.shape[0]}"
                )
            else:
                self.positions_by_index[int(index)] = pos
                self.velocities_by_index[int(index)] = vel
        dataset.close()
        self.source_indices = np.array(list(self.positions_by_index))
        self.resample_trajectories()
        return self.full_pos, self.full_vel, self.traj_lengths

    def resample_trajectories(self):
        """Draw a new recording assignment without reloading the dataset."""
        self.select_trajectories(self.rng.choice(self.source_indices, size=self.num_envs, replace=True))

    def select_trajectories(self, indices):
        indices = np.asarray(indices, dtype=int)
        if indices.shape != (self.num_envs,) or any(int(i) not in self.positions_by_index for i in indices):
            raise ValueError("Scenario recordings must belong to this environment's dataset partition")
        if hasattr(self, "trajectory_indices") and np.array_equal(indices, self.trajectory_indices):
            return
        self.trajectory_indices = indices.copy()
        pos_lst = [self.positions_by_index[int(i)] for i in indices]
        vel_lst = [self.velocities_by_index[int(i)] for i in indices]

        traj_lengths = [pos.shape[0] for pos in pos_lst]

        pos_traj = np.zeros((self.num_envs, max(traj_lengths), 3), dtype=np.float32)
        vel_traj = np.zeros((self.num_envs, max(traj_lengths), 3), dtype=np.float32)
        for i, (pos, vel) in enumerate(zip(pos_lst, vel_lst)):
            pos_traj[i, : pos.shape[0], :] = pos
            vel_traj[i, : vel.shape[0], :] = vel

        self.full_pos, self.full_vel, self.traj_lengths = pos_traj, vel_traj, traj_lengths


    def reset(self, idxs: np.ndarray | None = None) -> None:
        self._set_start_index(idxs)

    def step(self, active=None, **kwargs):
        if active is None:
            self.current_idx += 1
        else:
            self.current_idx[active] += 1

    def get_state(self, noise=None):
        env_idx = np.arange(self.num_envs)
        position = self.full_pos[env_idx, self.current_idx]
        velocity = self.full_vel[env_idx, self.current_idx]
        if noise is None:
            noise = self.rng.normal(0.0, self.noise_std, size=(2, self.num_envs, 3))
        noise_pos = position + noise[0]
        noise_vel = velocity + noise[1]
        
        return {
            "true_position": position,
            "true_velocity": velocity,
            "position": noise_pos,
            "velocity": noise_vel,
        }
