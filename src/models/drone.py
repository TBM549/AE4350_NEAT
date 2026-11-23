import numpy as np

from src.config.drone_config import drone_config


class vecDronePursuer_1order():
    def __init__(self, 
                 dt: float, 
                 num_envs: int, 
                 config: dict = drone_config, 
                 randomize_on_reset: bool = False, 
                 seed = 0):
        self.rng = np.random.default_rng(seed)
        self.num_envs = num_envs
        self.dt = dt
        self.randomize_on_reset = randomize_on_reset
        p_cfg = config

        self._load_config(p_cfg)
        self._randomized_parameters()

        # Noise parameters
        self.pos_noise_std = p_cfg["POSITION_NOISE_STD"]
        self.vel_noise_std = p_cfg["VELOCITY_NOISE_STD"]
        self.accel_noise_std = p_cfg["ACCELERATION_NOISE_STD"]

        # Vectorized state variables
        self.pos = np.zeros((num_envs, 3), dtype=np.float32)
        self.vel = np.zeros((num_envs, 3), dtype=np.float32)
        self.accel = np.zeros((num_envs, 3), dtype=np.float32)

        self.reset()
    
    def _load_config(self, p_cfg):
        self.tau_norm = p_cfg["ACTUATOR_TAU"]
        self.tau_pct = p_cfg["ACTUATOR_TAU_PCT"]
        self.max_accel_norm = p_cfg["MAX_ACCELERATION"]
        self.max_accel_pct = p_cfg["MAX_ACCELERATION_PCT"]
        self.max_speed_norm = p_cfg["MAX_SPEED"]
        self.max_speed_pct = p_cfg["MAX_SPEED_PCT"]

        self.init_pos_norm = np.array(p_cfg["INITIAL_POS"], dtype=np.float32)
        self.init_radius_max = p_cfg["INITIAL_RADIUS"]
        self.init_speed_max = p_cfg["INITIAL_SPEED"]
        self.init_accel_max = p_cfg["INITIAL_ACCELERATION"]

        self.tau = np.full(self.num_envs, self.tau_norm, dtype=np.float32)
        self.max_accel = np.full(self.num_envs, self.max_accel_norm, dtype=np.float32)
        self.max_speed = np.full(self.num_envs, self.max_speed_norm, dtype=np.float32)
        self.init_position = np.full((self.num_envs, 3), self.init_pos_norm, dtype=np.float32)
        self.init_velocity = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.init_acceleration = np.zeros((self.num_envs, 3), dtype=np.float32)

    def _sample_initial_positions(self):
        dirs = self.rng.normal(size=(self.num_envs, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = self.rng.uniform(0.0, self.init_radius_max, size=(self.num_envs, 1)).astype(np.float32)
        return self.init_pos_norm + radii * dirs

    def _sample_initial_velocities(self):
        dirs = self.rng.normal(size=(self.num_envs, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        speeds = self.rng.uniform(0.0, self.init_speed_max, size=(self.num_envs, 1)).astype(np.float32)
        return speeds * dirs

    def _sample_initial_accelerations(self):
        dirs = self.rng.normal(size=(self.num_envs, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        accels = self.rng.uniform(0.0, self.init_accel_max, size=(self.num_envs, 1)).astype(np.float32)
        return accels * dirs

    def _get_randomized(self, nominal: float, percentage: float) -> np.ndarray:
        return self.rng.uniform(1 - percentage, 1 + percentage, size=self.num_envs) * nominal

    def _randomized_parameters(self, idx=None):
        if idx is None:
            idx = np.arange(self.num_envs)
        self.tau[idx] = self._get_randomized(self.tau_norm, self.tau_pct)[idx]
        self.max_accel[idx] = self._get_randomized(self.max_accel_norm, self.max_accel_pct)[idx]
        self.max_speed[idx] = self._get_randomized(self.max_speed_norm, self.max_speed_pct)[idx]

        self.init_position[idx] = self._sample_initial_positions()[idx]
        self.init_velocity[idx] = self._sample_initial_velocities()[idx]
        self.init_acceleration[idx] = self._sample_initial_accelerations()[idx]

    def reset(self, idx = None) -> None:
        full_reset = idx is None
        if idx is None:
            idx = np.arange(self.num_envs)
        if self.randomize_on_reset:
            self._randomized_parameters(idx)

        if full_reset:
            self.pos = self.init_position.copy()
            self.vel = self.init_velocity.copy()
            self.accel = self.init_acceleration.copy()
        else:
            self.pos[idx] = self.init_position[idx]
            self.vel[idx] = self.init_velocity[idx]
            self.accel[idx] = self.init_acceleration[idx]

    def step(self, command) -> None:
        accel_cmd = command

        raw_cmd = np.clip(accel_cmd, -1, 1)
        mag_cmd = np.linalg.norm(raw_cmd, axis=1, keepdims=True)
        safe_raw_cmd = np.where(mag_cmd < 1, raw_cmd * self.max_accel[:, None],
                                raw_cmd / (mag_cmd + 1e-8) * self.max_accel[:, None])

        alpha = self.dt / (self.tau + self.dt)[:, None]
        self.accel = (1 - alpha) * self.accel + alpha * safe_raw_cmd

        self.vel += self.accel * self.dt
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_speed[:, None] / (speed + 1e-8))
        self.vel *= scale

        self.pos += self.vel * self.dt

    def get_state(self, noise=None) -> dict:
        position = self.pos.copy()
        velocity = self.vel.copy()
        acceleration = self.accel.copy()

        if noise is None:
            noise = self.rng.normal(size=(3, self.num_envs, 3))
            noise *= np.array([self.pos_noise_std, self.vel_noise_std, self.accel_noise_std])[:, None, None]
        noisy_pos = position + noise[0]
        noisy_vel = velocity + noise[1]
        noisy_accel = acceleration + noise[2]

        return {
            "true_position": position,
            "position": noisy_pos,
            "true_velocity": velocity,
            "velocity": noisy_vel,
            "true_acceleration": acceleration,
            "acceleration": noisy_accel,
        }
