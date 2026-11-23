"""Run all report evaluations and plot every baseline flight."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.evaluate_baseline import run as baseline
from src.evaluation.evaluate_sensitivity import run as sensitivity
from src.evaluation.plot.plot_baseline_flights import run as flight_plots



if __name__ == "__main__":
    baseline()
    sensitivity()
    flight_plots()
    print("Done. All outputs are in results/baseline and results/sensitivity.")
