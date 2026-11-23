from src.config.drone_config import drone_config
from src.config.moth_config import moth_config

env_config = {
    "DT": 0.01,
    "NUM_ENVS": 256,

    "MAX_TIMESTEPS": 300,
    "CATCH_DISTANCE": 0.05,
    "ATTEMPT_DISTANCE_MULTIPLIER": 5.0,  # 0.60 m attempt radius / 0.05 m catch distance

    "CATCH_REWARD": 100.0,
    "DISTANCE_WEIGHT": 5.0,
    "REWARD_DISTANCE_SCALE": 0.5,  # metres; independent of observation scaling
    "TIME_WEIGHT": 10.0,

    "DISTANCE_NORM": 2.0,
    "VELOCITY_NORM": 5.0,
    "ACCELERATION_NORM": 15.0,

    "INSECT_BOUNDARY_RADIUS": 10.0,

    "RANDOMIZE_ON_RESET": True,

    "PURSUER_CONFIG": drone_config,
    "EVADER_CONFIG": moth_config
}
