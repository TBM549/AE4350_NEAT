drone_config = {
    "INITIAL_POS": [2.5, 0.0, 1.0],
    "INITIAL_RADIUS": 0.25,
    "INITIAL_SPEED": 0.0,
    "INITIAL_ACCELERATION": 0.0,

    "MAX_ACCELERATION": 15,  # Maximum acceleration used for scaling acceleration commands
    "MAX_ACCELERATION_PCT": 0.0,  # Percentage of maximum acceleration for randomization
    
    "MAX_SPEED": 7.5,  # relevant only for 1st order acceleration model
    "MAX_SPEED_PCT": 0.0,  # Percentage of maximum speed for randomization

    "ACTUATOR_TAU": 0.05,  # relevant only for 1st order acceleration model
    "ACTUATOR_TAU_PCT": 0.0,  # Percentage of actuator time constant for randomization

    "POSITION_NOISE_STD": 0.0,
    "VELOCITY_NOISE_STD": 0.0,
    "ACCELERATION_NOISE_STD": 0.0
    }