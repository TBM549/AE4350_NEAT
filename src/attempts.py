"""Track close approaches without changing simulation state or fitness."""

import numpy as np


class AttemptTracker:
    def __init__(self, initial_distance, radius):
        self.previous = np.asarray(initial_distance, dtype=float).copy()
        self.radius = radius
        self.inside = self.previous <= radius
        self.open = self.inside.copy()
        self.closing = np.zeros(len(self.previous), dtype=bool)
        self.count = self.inside.astype(int)
        self.failed = np.zeros(len(self.previous), dtype=int)
        self.successful = np.zeros(len(self.previous), dtype=int)
        self.first = np.where(self.inside, 0., np.nan)
        self.start = np.zeros(len(self.previous))
        self.minimum = self.previous.copy()
        self.failed_distance = np.zeros(len(self.previous))
        self.failed_duration = np.zeros(len(self.previous))

    def update(self, distances, time, active, caught):
        distances = np.asarray(distances, dtype=float)
        time = np.broadcast_to(np.asarray(time), distances.shape)
        active = np.asarray(active, dtype=bool)
        caught = np.asarray(caught, dtype=bool) & active
        inside = distances <= self.radius
        # A catch after a confirmed miss is a separate recovery attempt.
        entered = active & ((inside & ~self.inside) | (caught & ~self.open))
        self.count[entered] += 1
        first = entered & np.isnan(self.first)
        self.first[first] = time[first]
        self.start[entered] = time[entered]
        self.minimum[entered] = distances[entered]
        self.open[entered] = True
        self.closing[entered] = False
        approaching = active & self.open
        self.closing[approaching] |= distances[approaching] < self.previous[approaching]
        self.minimum[approaching] = np.minimum(self.minimum[approaching], distances[approaching])
        failed = (approaching & self.closing & (distances > self.previous)
                  & self.inside & ~caught)
        self.failed[failed] += 1
        self.failed_distance[failed] += self.minimum[failed]
        self.failed_duration[failed] += time[failed] - self.start[failed]
        self.successful[caught] += 1
        self.open[failed | caught | (active & ~inside)] = False
        self.inside[active] = inside[active]
        self.previous[active] = distances[active]

    def episode_metrics(self):
        return [dict(attempts=int(self.count[i]), failed_attempts=int(self.failed[i]),
                     successful_attempts=int(self.successful[i]),
                     unfinished_attempts=int(self.count[i]-self.failed[i]-self.successful[i]),
                     first_attempt_time_s=float(self.first[i]) if np.isfinite(self.first[i]) else None,
                     mean_failed_attempt_distance_m=float(self.failed_distance[i]/self.failed[i])
                         if self.failed[i] else None,
                     mean_failed_attempt_duration_s=float(self.failed_duration[i]/self.failed[i])
                         if self.failed[i] else None)
                for i in range(len(self.count))]


def summarize_attempts(rows):
    totals = {name: sum(row[name] for row in rows) for name in
              ("attempts", "failed_attempts", "successful_attempts", "unfinished_attempts")}
    first = [row["first_attempt_time_s"] for row in rows if row["first_attempt_time_s"] is not None]
    result = {"total_"+name: int(value) for name, value in totals.items()}
    result.update(mean_attempts=totals["attempts"]/len(rows) if rows else 0.,
                  mean_failed_attempts=totals["failed_attempts"]/len(rows) if rows else 0.,
                  attempt_episode_rate=sum(row["attempts"] > 0 for row in rows)/len(rows) if rows else 0.,
                  mean_first_attempt_time_s=float(np.mean(first)) if first else None,
                  attempt_success_rate=totals["successful_attempts"]/totals["attempts"]
                      if totals["attempts"] else None)
    for metric in ("mean_failed_attempt_distance_m", "mean_failed_attempt_duration_s"):
        result[metric] = (sum(row[metric]*row["failed_attempts"] for row in rows
                              if row["failed_attempts"])/totals["failed_attempts"]
                          if totals["failed_attempts"] else None)
    return result
