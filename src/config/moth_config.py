moth_config = {
    "DATASET_PATH": "moth_dataset/moth_train_dataset.npz",
    "NOISE_STD": 0.00,  # noise added to positions only
    "FILTER_TYPE": "dataset",  # options: [dataset, passthrough, ekf (legacy/untested)]
    "FILTER_PARAMS": {
        "process_noise": 1e-4,  # ekf q_acc
        "measurement_noise": 1e-2,  # ekf r_pos
        "pos_window_samples": 10,
        "vel_window_samples": 5,
        "vel_from_filtered_pos": True,
    }
}
