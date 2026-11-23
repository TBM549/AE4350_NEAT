import numpy as np

from src.models.moth import VecMothEvader
from src.models.drone import vecDronePursuer_1order
from src.attempts import AttemptTracker

class Enviroment:
    def __init__(self, config, seed=0):
        self.config = config
        self.pursuer_config = config["PURSUER_CONFIG"]
        self.evader_config = config["EVADER_CONFIG"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scenarios = None

        self.max_timesteps = config["MAX_TIMESTEPS"]
        self.catch_distance = config["CATCH_DISTANCE"]
        self.attempt_distance_multiplier = config.get("ATTEMPT_DISTANCE_MULTIPLIER", 12.0)
        if not np.isfinite(self.attempt_distance_multiplier) or self.attempt_distance_multiplier < 1:
            raise ValueError("ATTEMPT_DISTANCE_MULTIPLIER must be finite and at least 1")
        self.attempt_distance = self.catch_distance * self.attempt_distance_multiplier
        
        self.catch_reward = config["CATCH_REWARD"]
        self.distance_weight = config["DISTANCE_WEIGHT"]
        self.reward_distance_scale = config.get("REWARD_DISTANCE_SCALE", 0.5)
        if not np.isfinite(self.reward_distance_scale) or self.reward_distance_scale <= 0:
            raise ValueError("REWARD_DISTANCE_SCALE must be finite and positive")
        self.time_weight = config["TIME_WEIGHT"]

        self.dt = config["DT"]
        self.num_envs = config["NUM_ENVS"]
        self.randomize_on_reset = config["RANDOMIZE_ON_RESET"]

        self.boundary_distance = np.array(config["INSECT_BOUNDARY_RADIUS"], dtype=np.float32)
        self.distance_norm = config["DISTANCE_NORM"]
        self.velocity_norm = config["VELOCITY_NORM"]
        self.acceleration_norm = config["ACCELERATION_NORM"]

        # initialize evader
        self.evader = VecMothEvader(self.dt, self.num_envs, self.max_timesteps, self.evader_config, self.randomize_on_reset, seed=seed)
        self.pursuer = vecDronePursuer_1order(self.dt, self.num_envs, self.pursuer_config, self.randomize_on_reset, seed=seed)

        self.reset()


    def sample_scenarios(self, resample_recordings=False):
        if resample_recordings:
            self.evader.resample_trajectories()
        self.reset()
        scenarios = {"start_idx": self.evader.current_idx.copy(),
                     "trajectory_indices": self.evader.trajectory_indices.copy()}
        for name in ("tau", "max_accel", "max_speed", "init_position", "init_velocity", "init_acceleration"):
            scenarios[name] = getattr(self.pursuer, name).copy()

        # Store observation noise by timestep so extra state reads do not change it.
        shape = (self.max_timesteps + 1, self.num_envs, 3)
        scenarios["evader_noise"] = self.rng.normal(0.0, self.evader.noise_std, size=(shape[0], 2, *shape[1:]))
        noise_std = np.array([self.pursuer.pos_noise_std, self.pursuer.vel_noise_std, self.pursuer.accel_noise_std])
        scenarios["pursuer_noise"] = self.rng.normal(size=(shape[0], 3, *shape[1:])) * noise_std[None, :, None, None]

        # Gather the shared moth motion once, including the initial observation.
        indices = scenarios["start_idx"][None, :] + np.arange(shape[0])[:, None]
        env_indices = np.arange(self.num_envs)[None, :]
        scenarios["evader_position"] = self.evader.full_pos[env_indices, indices]
        scenarios["evader_velocity"] = self.evader.full_vel[env_indices, indices]
        scenarios["evader_observed_position"] = scenarios["evader_position"] + scenarios["evader_noise"][:, 0]
        scenarios["evader_observed_velocity"] = scenarios["evader_velocity"] + scenarios["evader_noise"][:, 1]
        return scenarios

    def reset(self, idx=None, scenarios=None):
        if idx is not None and hasattr(self, "dones"):
            if scenarios is not None or self.scenarios is not None:
                raise ValueError("Partial reset is unavailable for a fixed scenario bank; use reset(scenarios=...).")
            indices = np.arange(self.num_envs)[idx]
            indices = np.atleast_1d(indices)
            self.evader.reset(indices)
            self.pursuer.reset(indices)
            self.episode_steps[indices] = 0
            self.dones[indices] = False
            self.caught[indices] = False
            self.catch_times[indices] = np.nan
            self.fitness_step[indices] = 0
            self.command[indices] = 0
            target = self.evader.full_pos[indices, self.evader.current_idx[indices]]
            distance = np.linalg.norm(target - self.pursuer.pos[indices], axis=1)
            self.minimum_distance[indices] = distance
            fresh = AttemptTracker(distance, self.attempt_distance)
            for name, value in vars(fresh).items():
                if isinstance(value, np.ndarray):
                    getattr(self.attempts, name)[indices] = value
            self.fitness[indices] = self.calculate_fitness()[indices]
            return self.get_observations(), self.dones
        self.scenarios = scenarios
        if scenarios is None:
            self.evader.reset(idx)
            self.pursuer.reset(idx)
        else:
            if "trajectory_indices" in scenarios:
                self.evader.select_trajectories(scenarios["trajectory_indices"])
            self.evader.current_idx = scenarios["start_idx"].copy()
            for name in ("tau", "max_accel", "max_speed", "init_position", "init_velocity", "init_acceleration"):
                setattr(self.pursuer, name, scenarios[name].copy())
            self.pursuer.pos = self.pursuer.init_position.copy()
            self.pursuer.vel = self.pursuer.init_velocity.copy()
            self.pursuer.accel = self.pursuer.init_acceleration.copy()

        self.timestep = 0
        self.episode_steps = np.zeros(self.num_envs, dtype=int)
        self.command = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.dones = np.zeros(self.num_envs, dtype=bool)
        self.caught = np.zeros(self.num_envs, dtype=bool)
        self.catch_times = np.full(self.num_envs, np.nan)
        self.fitness_step = np.zeros(self.num_envs, dtype=np.float32)

        initial_target = (self.evader.full_pos[np.arange(self.num_envs), self.evader.current_idx]
                          if scenarios is None else scenarios["evader_position"][0])
        self.minimum_distance = np.linalg.norm(initial_target - self.pursuer.pos, axis=1)
        self.attempts = AttemptTracker(self.minimum_distance, self.attempt_distance)
        self.fitness = self.calculate_fitness()

        observations = self.get_observations()

        return observations, self.dones

    def step(self, accel_cmd):
        if np.all(self.dones):
            self.fitness_step.fill(0)
            return self.get_observations(), self.dones
        active = ~self.dones.copy()
        previous_fitness = self.fitness.copy()
        self.command = np.asarray(accel_cmd).copy()
        terminal_states = {name: getattr(self.pursuer, name)[~active].copy()
                           for name in ("pos", "vel", "accel")}
        self.pursuer.step(accel_cmd)
        for name, values in terminal_states.items():
            getattr(self.pursuer, name)[~active] = values
        self.evader.step(active=active)

        pursuer_pos = self.pursuer.pos
        if self.scenarios is None:
            evader_pos = self.evader.full_pos[np.arange(self.num_envs), self.evader.current_idx]
        else:
            evader_pos = self.scenarios["evader_position"][self.timestep + 1]

        # compute distance to target
        dist = np.linalg.norm(pursuer_pos - evader_pos, axis=1)
        inside_boundary = dist <= self.boundary_distance 
        self.minimum_distance[active] = np.minimum(self.minimum_distance[active], dist[active])

        # update dones and caught        
        new_catches = (~self.dones) & (dist < self.catch_distance)
        episode_time = (self.episode_steps + 1) * self.dt
        self.catch_times[new_catches] = episode_time[new_catches]
        self.caught |= new_catches
        self.attempts.update(dist, episode_time, active, new_catches)

        self.dones = self.get_dones(dist, inside_boundary)
        self.episode_steps[active] += 1
        self.fitness = self.calculate_fitness()
        self.fitness_step = self.fitness - previous_fitness

        self.timestep += 1
        observations = self.get_observations()
        return observations, self.dones

    def get_dones(self, dist, inside_boundary):
        # done if distance is less than catch distance or max timesteps reached
        # ones done keep done until reset
        done_boundary = ~inside_boundary
        done_catch = dist < self.catch_distance        
        done_time = (self.episode_steps + 1) >= self.max_timesteps
        dones = self.dones | done_boundary | done_catch | done_time
        return dones

    def vectorize_state(self, state, norm_factor):
        magnitude = np.linalg.norm(state, axis=1, keepdims=True)
        safe_magnitude = np.where(magnitude > 1e-8, magnitude, 1.0)
        direction = np.where(magnitude > 1e-8, state / safe_magnitude, 0.0)
        return np.concatenate([magnitude / norm_factor, direction], axis=1)

    def get_observations(self):
        # observations are the relative position of the evader to the pursuer
        pursuer_noise = None if self.scenarios is None else self.scenarios["pursuer_noise"][self.timestep]
        pursuer_states = self.pursuer.get_state(noise=pursuer_noise)
        if self.scenarios is None:
            evader_states = self.evader.get_state()
            evader_pos = evader_states["position"]
            evader_vel = evader_states["velocity"]
        else:
            evader_pos = self.scenarios["evader_observed_position"][self.timestep]
            evader_vel = self.scenarios["evader_observed_velocity"][self.timestep]
        relative_pos = evader_pos - pursuer_states["position"]
        relative_vel = evader_vel - pursuer_states["velocity"]
        accel = pursuer_states["acceleration"]
        acceleration = accel / self.acceleration_norm
        return np.concatenate([self.vectorize_state(relative_pos, self.distance_norm),
                                   self.vectorize_state(relative_vel, self.velocity_norm),
                                   acceleration], axis=1)

    def calculate_fitness(self):
        score = self.distance_weight * np.exp(-self.minimum_distance / self.reward_distance_scale)
        score += np.where(np.isnan(self.catch_times), 0.0, self.catch_reward \
                          - self.time_weight * self.catch_times / (self.max_timesteps * self.dt))
        return score

    def get_states(self):
        pursuer_states = self.pursuer.get_state()
        evader_states = self.evader.get_state()
        return {
            "pursuer": pursuer_states,
            "evader": evader_states,
            "command": self.command.copy(),
            "timestep": self.timestep,
            "fitness": self.fitness.copy(),
            "fitness_step": self.fitness_step.copy(),
            "dones": self.dones.copy(),
        }
    
    
