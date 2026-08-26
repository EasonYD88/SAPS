from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    ci_low: float
    ci_high: float


def paired_bootstrap(
    frame: pd.DataFrame,
    method: str,
    baseline: str,
    value_column: str,
    replicates: int,
    seed: int,
) -> BootstrapInterval:
    pivot = frame.pivot_table(
        index="example_id", columns="scheduler", values=value_column, aggfunc="mean"
    ).dropna(subset=[method, baseline])
    differences = (pivot[method] - pivot[baseline]).to_numpy()
    if not len(differences):
        return BootstrapInterval(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(replicates, len(differences)), replace=True)
    means = samples.mean(axis=1)
    return BootstrapInterval(
        float(differences.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )

