# Load, plot, and save the moth dataset after preprocessing.

import os
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resample_trajectory(times, *states, dt=0.01):
    """Interpolate sample-aligned states onto a uniform clock, without extrapolation."""
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("Trajectory needs at least two finite timestamps")
    times = times - times[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    count = int(np.floor(times[-1] / dt + 1e-10)) + 1
    clock = np.arange(count, dtype=np.float64) * dt
    result = []
    for state in states:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (len(times), 3) or not np.all(np.isfinite(state)):
            raise ValueError("Trajectory states must be finite arrays of shape (samples, 3)")
        result.append(np.column_stack([
            np.interp(clock, times, state[:, axis]) for axis in range(3)
        ]))
    return clock, *result


def pats_to_ned(arr: np.ndarray) -> np.ndarray:
    """
    Convert an (..., 3) array from PATS to NED.
    PATS: left-handed, x-left, y-up, z-backward
    NED: right-handed, x-north, y-east, z-down
    Returns new array with the same shape as input.
    """
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.stack([-z, -x, -y], axis=-1)


class VecMothLoader():
    def __init__(
        self,
        moth_folder: str | Path,
        ekf_process_noise: float = 1e-4,
        ekf_measurement_noise: float = 5e-2,
        min_trajectory_length: int = 500,
        seed: int = 0,
        dt: float = 0.01,
    ):
        np.random.seed(seed)
        self.folder_path = moth_folder

        self.q_acc = ekf_process_noise
        self.r_pos = ekf_measurement_noise
        self.min_traj_len = min_trajectory_length
        self.dt = dt

        self._load_and_preprocess_all()

    def _load_and_preprocess_all(self):
        trajs = []
        times_list = []
        vel_trajs = []
        filenames = []

        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith(".csv"):
                path = os.path.join(self.folder_path, file)
                try:
                    df = pd.read_csv(path, sep=";").copy()
                    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
                    times = df["elapsed"].to_numpy(dtype=np.float64, copy=True)
                    times -= times[0]

                    pos = (
                        df[["sposX_insect", "sposY_insect", "sposZ_insect"]]
                        .astype(float)
                        .values
                    )
                    pos = pats_to_ned(pos)

                    vel = (
                        df[["svelX_insect", "svelY_insect", "svelZ_insect"]]
                        .astype(float)
                        .values
                    )

                    vel = pats_to_ned(vel)

                    trajs.append(pos)
                    vel_trajs.append(vel)
                    times_list.append(times)
                    filenames.append(file)

                except Exception as e:
                    print(f"[Error] Failed to load {file}: {e}")

        if not trajs:
            raise ValueError("No valid moth trajectories found.")

        num_trajectories = len(trajs)

        self.true_pos = []
        self.true_vel = []
        self.filtered_pos = []
        self.filtered_vel = []
        self.times = []
        self.true_times = []

        self.range_size = []
        self.trajectory_lengths = []
        self.filenames = []

        for i in range(num_trajectories):
            pos = trajs[i].astype(np.float32, copy=False)
            vel = vel_trajs[i].astype(np.float32, copy=False)
            times = times_list[i].astype(np.float64, copy=False)

            f_pos, f_vel = self._apply_ekf(pos, times)
            times, pos, vel, f_pos, f_vel = resample_trajectory(
                times, pos, vel, f_pos, f_vel, dt=self.dt
            )

            if len(f_pos) < self.min_traj_len:
                print(f"[Warning] Trajectory {filenames[i]} is too short ({len(f_pos)} points), skipping.")
                continue
            count = 0
            while len(f_pos) >= self.min_traj_len * 2:
                f_pos_1 = f_pos[:self.min_traj_len]
                f_vel_1 = f_vel[:self.min_traj_len]
                times_1 = times[:self.min_traj_len]

                pos_1 = pos[:self.min_traj_len]
                vel_1 = vel[:self.min_traj_len]
                
                self.true_pos.append(pos_1)
                self.true_vel.append(vel_1)

                self.filtered_pos.append(f_pos_1)
                self.filtered_vel.append(f_vel_1)
                self.times.append(times_1)
                self.trajectory_lengths.append(self._path_length(f_pos_1))
                range_size = self._flight_span(f_pos_1)
                self.range_size.append(range_size)
                self.filenames.append(f"{filenames[i]}_part{count}")

                f_pos = f_pos[self.min_traj_len:]
                f_vel = f_vel[self.min_traj_len:]
                times = times[self.min_traj_len:]

                pos = pos[self.min_traj_len:]
                vel = vel[self.min_traj_len:]
                count += 1

            self.filtered_pos.append(f_pos)
            self.filtered_vel.append(f_vel)
            self.times.append(times)

            self.true_pos.append(pos)
            self.true_vel.append(vel)

            range_size = self._flight_span(f_pos)
            self.range_size.append(range_size)
            self.trajectory_lengths.append(self._path_length(f_pos))
            self.filenames.append(f"{filenames[i]}" if count == 0 else f"{filenames[i]}_part{count}")

        self.num_trajectories = len(self.filtered_pos)
        self.trajectory_lengths = np.array(self.trajectory_lengths, dtype=np.float32)
        self.range_size = np.array(self.range_size, dtype=np.float32)

    def _apply_ekf(self, z_meas, times):
        N = len(z_meas)
        dt_seq = np.diff(times, prepend=times[0])
        x_hat = np.zeros((N, 3, 3))  # axis, state_dim, time
        pos_est = np.zeros((N, 3))
        vel_est = np.zeros((N, 3))

        for axis in range(3):
            x = np.zeros((3, N))
            P = np.eye(3)
            F = np.eye(3)
            Q = np.diag([0.0, 0.0, self.q_acc])
            H = np.array([[1, 0, 0]])
            R = np.array([[self.r_pos]])

            x[:, 0] = [z_meas[0, axis], 0.0, 0.0]
            for k in range(1, N):
                dt = dt_seq[k]
                F = np.array(
                    [
                        [1, dt, 0.5 * dt**2],
                        [0, 1, dt],
                        [0, 0, 1],
                    ]
                )
                # Predict
                x_pred = F @ x[:, k - 1]
                P_pred = F @ P @ F.T + Q

                # Update
                y = z_meas[k, axis] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x[:, k] = x_pred + (K @ y).flatten()
                P = (np.eye(3) - K @ H) @ P_pred

            pos_est[:, axis] = x[0]
            vel_est[:, axis] = x[1]

        return pos_est, vel_est


    def _flight_span(self, path: np.ndarray):
        position_range = np.array([[ path[:, 0].min(), path[:, 0].max()], 
                                   [path[:, 1].min(), path[:, 1].max()], 
                                   [path[:, 2].min(), path[:, 2].max()]], dtype=np.float32)
        range_size = float(np.linalg.norm(position_range[:, 1] - position_range[:, 0]))
        return range_size

    def _path_length(self, path: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def save_dataset(self, output_path: str | Path):
        return self.save_dataset_split(output_path, np.arange(self.num_trajectories))

    def save_dataset_split(self, output_path: str | Path, indices: np.ndarray):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        indices = np.asarray(indices, dtype=np.int32)
        np.savez_compressed(
            output_path,
            true_pos=np.array([self.true_pos[i] for i in indices], dtype=object),
            true_vel=np.array([self.true_vel[i] for i in indices], dtype=object),
            filtered_pos=np.array([self.filtered_pos[i] for i in indices], dtype=object),
            filtered_vel=np.array([self.filtered_vel[i] for i in indices], dtype=object),
            times=np.array([self.times[i] for i in indices], dtype=object),
            filenames=np.array([self.filenames[i] for i in indices], dtype=object),
            trajectory_lengths=np.array([self.trajectory_lengths[i] for i in indices], dtype=np.int32),
            range_size=np.array([self.range_size[i] for i in indices], dtype=np.float32),
            indices=indices,
            sample_dt=np.float64(self.dt),
        )
        return output_path

    def _get_split_indices(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        if indices is None:
            return np.arange(self.num_trajectories, dtype=np.int32)
        return np.asarray(indices, dtype=np.int32)

    def _get_axis_limits(self, indices: np.ndarray | list[int] | None = None):
        indices = self._get_split_indices(indices)
        points = np.concatenate([self.filtered_pos[i] for i in indices], axis=0)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = maxs - mins
        padding = np.maximum(spans * 0.05, 1e-3)
        return mins - padding, maxs + padding

    def _set_axis_limits(self, ax, mins: np.ndarray, maxs: np.ndarray):
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_zlim(float(mins[2]), float(maxs[2]))

    def _subplot_grid(self, num_plots: int):
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        return rows, cols

    def plot_trajectories(
        self,
        output_path: str | Path | None = None,
        max_trajs: int | None = None,
        subplot_output_path: str | Path | None = None,
        valid_only: bool = True,
        indices: np.ndarray | list[int] | None = None,
        title_prefix: str = "Moth",
    ):
        del max_trajs, valid_only
        indices = self._get_split_indices(indices)
        mins, maxs = self._get_axis_limits(indices)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for idx in indices:
            path = self.filtered_pos[idx]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.8, linewidth=1.2)

        start_points = np.array([self.filtered_pos[idx][0] for idx in indices])
        ax.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2], color="green", s=18, label="Start")
        ax.set_title(f"{title_prefix} flight paths ({len(indices)} trajectories)")
        ax.set_xlabel("North")
        ax.set_ylabel("East")
        ax.set_zlabel("Down")
        ax.zaxis.set_inverted(True)  # Invert z-axis for NED
        self._set_axis_limits(ax, mins, maxs)
        ax.legend()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        rows, cols = self._subplot_grid(len(indices))
        subplot_fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
        subplot_axes = []
        for plot_idx, idx in enumerate(indices):
            path = self.filtered_pos[idx]
            name = self.filenames[idx]
            traj_length = self.trajectory_lengths[idx]

            traj_ax = subplot_fig.add_subplot(rows, cols, plot_idx + 1, projection="3d")
            traj_ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=1.5, color="tab:blue")
            traj_ax.scatter(*path[0], color="green", s=30, label="Start")
            traj_ax.scatter(*path[-1], color="red", s=30, label="End")
            traj_ax.set_title(
                f"Trajectory {idx} / {name}\npoints={len(path)} | path={traj_length:.3f} | span={self.range_size[idx]:.3f}"
            )
            traj_ax.set_xlabel("North")
            traj_ax.set_ylabel("East")
            traj_ax.set_zlabel("Down")
            traj_ax.zaxis.set_inverted(True)  # Invert z-axis for NED
            self._set_axis_limits(traj_ax, mins, maxs)
            traj_ax.legend()
            subplot_axes.append(traj_ax)

        subplot_fig.suptitle(f"Individual {title_prefix.lower()} trajectories", fontsize=14)
        subplot_fig.tight_layout()

        if subplot_output_path is not None:
            subplot_output_path = Path(subplot_output_path)
            subplot_output_path.parent.mkdir(parents=True, exist_ok=True)
            subplot_fig.savefig(subplot_output_path, dpi=200, bbox_inches="tight")

        return fig, ax, subplot_fig, subplot_axes

    def plot_summary_stats(
        self,
        output_path: str | Path | None = None,
        indices: np.ndarray | list[int] | None = None,
        title_prefix: str = "Moth",
    ):
        indices = self._get_split_indices(indices)

        selected_vel = [self.filtered_vel[i] for i in indices if len(self.filtered_vel[i]) > 0]
        selected_pos = [self.filtered_pos[i] for i in indices if len(self.filtered_pos[i]) > 0]
        selected_step_distances = [np.linalg.norm(np.diff(self.filtered_pos[i], axis=0), axis=1) for i in indices if len(self.filtered_pos[i]) > 1]

        all_speeds = np.concatenate([np.linalg.norm(vel, axis=1) for vel in selected_vel])
        all_step_distances = np.concatenate(selected_step_distances)

        all_x_vels = np.concatenate([vel[:, 0] for vel in selected_vel])
        all_y_vels = np.concatenate([vel[:, 1] for vel in selected_vel])
        all_z_vels = np.concatenate([vel[:, 2] for vel in selected_vel])

        all_x_pos = np.concatenate([pos[:, 0] for pos in selected_pos])
        all_y_pos = np.concatenate([pos[:, 1] for pos in selected_pos])
        all_z_pos = np.concatenate([pos[:, 2] for pos in selected_pos])

        path_lengths = self.trajectory_lengths[indices]
        range_size = self.range_size[indices]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes[0, 0].hist(all_step_distances, bins=30, color="tab:orange", alpha=0.85)
        axes[0, 0].set_title("Step distance per trajectory")
        axes[0, 0].set_xlabel("Distance between consecutive points")
        axes[0, 0].set_ylabel("Count")

        axes[1, 0].hist(all_x_pos, bins=30, color="tab:blue", alpha=0.85, label="X - North")
        axes[1, 0].set_title("Positions Distribution")
        axes[1, 0].set_xlabel("Position")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].hist(all_y_pos, bins=30, color="tab:orange", alpha=0.85, label="Y - East")
        axes[1, 0].hist(all_z_pos, bins=30, color="tab:green", alpha=0.85, label="Z - Down")
        axes[1, 0].legend()

        axes[0, 1].hist(all_speeds, bins=40, color="tab:blue", alpha=0.85)
        axes[0, 1].set_title("Velocity magnitudes")
        axes[0, 1].set_xlabel("Speed")
        axes[0, 1].set_ylabel("Count")

        axes[1, 1].hist(all_x_vels, bins=30, color="tab:blue", alpha=0.85, label="X - North")
        axes[1, 1].set_title("Velocity components")
        axes[1, 1].set_xlabel("Velocity")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].hist(all_y_vels, bins=30, color="tab:orange", alpha=0.85, label="Y - East")
        axes[1, 1].hist(all_z_vels, bins=30, color="tab:green", alpha=0.85, label="Z - Down")
        axes[1, 1].legend()

        axes[0, 2].hist(path_lengths, bins=30, color="tab:green", alpha=0.85)
        axes[0, 2].set_title("Trajectory path lengths")
        axes[0, 2].set_xlabel("Total path length")
        axes[0, 2].set_ylabel("Count")

        axes[1, 2].hist(range_size, color="tab:red", alpha=0.8)
        axes[1, 2].set_title("Trajectory flight span")
        axes[1, 2].set_xlabel("Flight span size (distance between min and max positions)")
        axes[1, 2].set_ylabel("Count")

        fig.suptitle(f"{title_prefix} summary statistics", fontsize=14)
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        return fig, axes

    def get_train_eval_indices(self, eval_ratio: float = 0.2, split="recording"):
        """Keep sources together by default; retain the historical split explicitly.

        legacy: first N segments, reproducing the supplied data with ratio 0.16
        and the original raw recordings. Filtering and segmentation are unchanged.
        """
        if not 0.0 < eval_ratio < 1.0:
            raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}.")

        if split == "legacy":
            num_eval = max(1, int(np.floor(self.num_trajectories * eval_ratio)))
            eval_indices = np.arange(num_eval, dtype=np.int32)
            train_indices = np.arange(num_eval, self.num_trajectories, dtype=np.int32)
        elif split == "recording":
            groups = [re.sub(r"_part\d+$", "", name) for name in self.filenames]
            sources = list(dict.fromkeys(groups))
            if len(sources) < 2:
                raise ValueError("At least two source recordings are needed for a recording split")
            num_eval = max(1, min(len(sources) - 1, int(np.floor(len(sources) * eval_ratio))))
            held_out = np.isin(groups, sources[:num_eval])
            eval_indices = np.flatnonzero(held_out).astype(np.int32)
            train_indices = np.flatnonzero(~held_out).astype(np.int32)
        else:
            raise ValueError(f"Unknown dataset split: {split}")

        if len(train_indices) == 0:
            raise ValueError(
                f"eval_ratio={eval_ratio} leaves no training trajectories. "
                f"Found {self.num_trajectories} total trajectories."
            )

        return train_indices, eval_indices




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/Users/tamar/OneDrive - Delft University of Technology/Thesis/MerlijnData/evader_datasets/opogona_old/top_moths"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("recording", "legacy"), default="recording")
    parser.add_argument("--eval-ratio", type=float, default=0.16)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    moth_folder = args.source
    eval_ratio = args.eval_ratio

    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or project_root / "moth_dataset"
    train_dataset_path = output_dir / "moth_train_dataset.npz"
    eval_dataset_path = output_dir / "moth_eval_dataset.npz"

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_plot_path = plot_dir / "moth_train_flight_paths.png"
    train_subplot_path = plot_dir / "moth_train_flight_subplots.png"
    train_summary_plot_path = plot_dir / "moth_train_summary_stats.png"

    eval_plot_path = plot_dir / "moth_eval_flight_paths.png"
    eval_subplot_path = plot_dir / "moth_eval_flight_subplots.png"
    eval_summary_plot_path = plot_dir / "moth_eval_summary_stats.png"

    loader = VecMothLoader(moth_folder=moth_folder, ekf_process_noise=1e-4, ekf_measurement_noise=5e-2, min_trajectory_length=650, seed=0)
    train_indices, eval_indices = loader.get_train_eval_indices(eval_ratio=eval_ratio, split=args.split)

    loader.save_dataset_split(train_dataset_path, train_indices)
    print(f"Saved training dataset to: {train_dataset_path}, {len(train_indices)} trajectories")
    loader.save_dataset_split(eval_dataset_path, eval_indices)
    print(f"Saved eval dataset to: {eval_dataset_path}, {len(eval_indices)} trajectories")

    if args.no_plots:
        raise SystemExit(0)

    train_fig, _, train_subplot_fig, _ = loader.plot_trajectories(
        output_path=train_plot_path,
        subplot_output_path=train_subplot_path,
        indices=train_indices,
        title_prefix="Training moth",
    )
    print(f"Saved training trajectory plot to: {train_plot_path}")
    print(f"Saved training subplot figure to: {train_subplot_path}")

    train_summary_fig, _ = loader.plot_summary_stats(
        output_path=train_summary_plot_path,
        indices=train_indices,
        title_prefix="Training moth",
    )
    print(f"Saved training summary plots to: {train_summary_plot_path}")

    eval_fig, _, eval_subplot_fig, _ = loader.plot_trajectories(
        output_path=eval_plot_path,
        subplot_output_path=eval_subplot_path,
        indices=eval_indices,
        title_prefix="Eval moth",
    )
    print(f"Saved eval trajectory plot to: {eval_plot_path}")
    print(f"Saved eval subplot figure to: {eval_subplot_path}")

    eval_summary_fig, _ = loader.plot_summary_stats(
        output_path=eval_summary_plot_path,
        indices=eval_indices,
        title_prefix="Eval moth",
    )
    print(f"Saved eval summary plots to: {eval_summary_plot_path}")

    # plt.show()
