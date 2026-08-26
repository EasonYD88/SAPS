from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_figures(frame: pd.DataFrame, run_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    specs = (
        ("projection_vs_gain", "gamma_pinv", "terminal_gain", "Projection vs gain"),
        ("planner_value_vs_cost", "planning_time_sec", "terminal_gain", "Value vs cost"),
        ("finite_budget_crossover", "epsilon_target", "terminal_gain", "Finite-budget crossover"),
        ("latency_frontier", "num_rounds", "response_power_direct", "Latency frontier"),
    )
    for name, x, y, title in specs:
        figure, axis = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=frame, x=x, y=y, hue="scheduler", marker="o", ax=axis)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(run_dir / f"{name}.png", dpi=200)
        figure.savefig(run_dir / f"{name}.pdf")
        plt.close(figure)

