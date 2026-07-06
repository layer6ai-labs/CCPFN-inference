"""
Eval datasets loaded from semi-synthetic CSV files (real covariates, synthetic
treatment and outcome from a hand-crafted DGP).

SemiSyntheticEvalDataset: CSV columns x_*, t, y, t_test, cepo_test. Expects
numeric covariate columns; NaNs are median-imputed on load.

SemiSyntheticPolicyEvalDataset: CSV columns x_*, t, y, t_optim, cepo_optim.
Covariate columns may be non-numeric (e.g. Lenta's `gender`). The loader keeps
`self.df` raw (dtypes and NaNs) so the paired `Scenario.dose_response` oracle
sees identical inputs to what the generation script saw, and builds a separate
fully-numeric `_df_numeric` twin for the prescriber (category-encoded strings,
median-imputed numerics).

When n_tables=1 the entire dataset is used with a fixed train/test split.
When n_tables>1 each table draws a reproducible random subsample of
subsample_size rows (defaulting to the full dataset size), giving different
train/test splits across tables for more robust metric estimates.
"""

import numpy as np
import pandas as pd

from .base import EvalDatasetCatalog, CEPO_Dataset, DRC_Dataset, DP_Dataset
from .scenarios.base import Scenario


class SemiSyntheticEvalDataset(EvalDatasetCatalog):
    def __init__(
        self,
        csv_path: str,
        n_tables: int = 1,
        test_ratio: float = 0.2,
        subsample_size: int | None = None,
        seed: int = 42,
    ):
        super().__init__(n_tables=n_tables, name="semi_synthetic")

        self.df = pd.read_csv(csv_path)
        _reserved = {"t", "y", "t_test", "cepo_test"}
        self.x_cols = [c for c in self.df.columns if c not in _reserved]
        self.df[self.x_cols] = self.df[self.x_cols].fillna(self.df[self.x_cols].median())
        self.test_ratio = test_ratio
        self.subsample_size = subsample_size if subsample_size is not None else len(self.df)
        self.seed = seed
        self.n_rows = len(self.df)

    def __getitem__(self, index: int) -> tuple[CEPO_Dataset, DRC_Dataset]:
        if index >= self.n_tables:
            raise IndexError(f"Index {index} out of range for {self.n_tables} tables")

        if self.n_tables == 1:
            # No randomness: use rows in original order, split at fixed boundary
            indices = np.arange(self.n_rows)
        else:
            # Draw a reproducible subsample for this table
            rng = np.random.RandomState(self.seed + index)
            indices = rng.choice(self.n_rows, size=min(self.subsample_size, self.n_rows), replace=False)
            indices = rng.permutation(indices)

        split = int(len(indices) * (1 - self.test_ratio))
        train_idx = indices[:split]
        test_idx = indices[split:]

        train_df = self.df.iloc[train_idx]
        test_df = self.df.iloc[test_idx]

        cepo_dataset = CEPO_Dataset(
            X_train=train_df[self.x_cols].to_numpy(dtype=float),
            t_train=train_df["t"].to_numpy(dtype=float),
            y_train=train_df["y"].to_numpy(dtype=float),
            X_test=test_df[self.x_cols].to_numpy(dtype=float),
            t_test=test_df["t_test"].to_numpy(dtype=float).reshape(-1, 1),
            true_cepo=test_df["cepo_test"].to_numpy(dtype=float).reshape(-1, 1),
        )
        drc_dataset = DRC_Dataset(
            X_train=train_df[self.x_cols].to_numpy(dtype=float),
            t_train=train_df["t"].to_numpy(dtype=float),
            y_train=train_df["y"].to_numpy(dtype=float),
            true_drc_t_y=[],
        )
        return cepo_dataset, drc_dataset


def _to_numeric_column(col: pd.Series) -> pd.Series:
    """Map a covariate column to a finite float Series: string/object columns
    are category-encoded, NaNs (including the -1 codes for originally-NaN
    strings) are filled with the column median, or 0.0 if the column is
    entirely NaN."""
    if pd.api.types.is_numeric_dtype(col):
        col = col.astype(float)
    else:
        col = col.astype("category").cat.codes.astype(float)
        col[col < 0] = np.nan
    median = col.median()
    return col.fillna(0.0 if pd.isna(median) else median)


class SemiSyntheticPolicyEvalDataset(EvalDatasetCatalog):
    """
    Policy-style eval dataset loaded from a semi-synthetic CSV paired with the
    live `Scenario` that generated it.

    CSV format: x_*, t, y, t_optim, cepo_optim. `t_optim`/`cepo_optim` are the
    per-row optimal treatment and the dose-response evaluated at it.

    The scenario is held alongside the CSV so the eval callback can call
    `scenario.dose_response` as an oracle when computing regret. `t_min`,
    `t_max`, and `lower_is_better` default to the scenario's; `lower_is_better`
    may be overridden explicitly.
    """

    def __init__(
        self,
        scenario: Scenario,
        csv_path: str,
        n_tables: int = 1,
        test_ratio: float = 0.2,
        subsample_size: int | None = None,
        seed: int = 42,
        lower_is_better: bool | None = None,
    ):
        super().__init__(n_tables=n_tables, name="semi_synthetic_policy")

        self.scenario = scenario
        self.t_min = scenario.t_min
        self.t_max = scenario.t_max
        self.lower_is_better = (
            lower_is_better if lower_is_better is not None else scenario.low_outcome_is_better
        )

        self.df = pd.read_csv(csv_path)
        self.x_cols = [c for c in self.df.columns if c not in {"t", "y", "t_optim", "cepo_optim"}]
        self.test_ratio = test_ratio
        self.subsample_size = subsample_size if subsample_size is not None else len(self.df)
        self.seed = seed
        self.n_rows = len(self.df)
        self._response_curve_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # self.df stays raw (dtypes + NaNs) so the scenario oracle sees the
        # exact inputs it saw at generation time — imputing would shift the
        # batch-standardization stats inside scenarios like Lenta and break
        # parity with the stored cepo_optim. The prescriber instead consumes
        # _df_numeric, a parallel fully-numeric twin.
        self._df_numeric = pd.DataFrame(
            {c: _to_numeric_column(self.df[c]) for c in self.x_cols},
            index=self.df.index,
        )

    def _split_indices(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic train/test row-index split for table `index`."""
        if index >= self.n_tables:
            raise IndexError(f"Index {index} out of range for {self.n_tables} tables")
        if self.n_tables == 1:
            indices = np.arange(self.n_rows)
        else:
            rng = np.random.RandomState(self.seed + index)
            indices = rng.choice(self.n_rows, size=min(self.subsample_size, self.n_rows), replace=False)
            indices = rng.permutation(indices)
        split = int(len(indices) * (1 - self.test_ratio))
        return indices[:split], indices[split:]

    def get_test_indices(self, index: int) -> np.ndarray:
        """Test-row indices into `self.df` for table `index`. The eval callback
        uses these to slice oracle outputs evaluated on the full dataframe
        (avoids batch-standardization drift inside scenario.dose_response)."""
        return self._split_indices(index)[1]
    
    def get_mean_response_curve(
        self,
        index: int,
        n_grid: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the oracle mean response curve for table `index`, averaged over
        the held-out test rows on a shared treatment grid.

        The full per-row oracle curve is generated once per `n_grid` via the
        scenario's `generate_response_curve(...)` helper and cached for reuse
        across evaluation epochs.
        """
        if n_grid not in self._response_curve_cache:
            t_grid = np.linspace(self.t_min, self.t_max, n_grid)
            full_x_df = self.df[self.x_cols]
            true_curve = np.stack(
                [self.scenario.dose_response(full_x_df, np.full(len(self.df), t)) for t in t_grid],
                axis=1,
            )
            self._response_curve_cache[n_grid] = (
                np.asarray(t_grid, dtype=float),
                np.asarray(true_curve, dtype=float),
            )

        t_grid, true_curve = self._response_curve_cache[n_grid]
        test_idx = self.get_test_indices(index)
        return t_grid, true_curve[test_idx].mean(axis=0)

    def get_response_curve(
        self,
        index: int,
        n_grid: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the oracle individual-level response curves for the held-out test
        rows in table `index` on a shared treatment grid.

        The full per-row oracle curve is generated once per `n_grid` via the
        scenario's `generate_response_curve(...)` helper and cached for reuse
        across evaluation epochs.
        """
        if n_grid not in self._response_curve_cache:
            t_grid = np.linspace(self.t_min, self.t_max, n_grid)
            full_x_df = self.df[self.x_cols]
            true_curve = np.stack(
                [self.scenario.dose_response(full_x_df, np.full(len(self.df), t)) for t in t_grid],
                axis=1,
            )
            self._response_curve_cache[n_grid] = (
                np.asarray(t_grid, dtype=float),
                np.asarray(true_curve, dtype=float),
            )

        t_grid, true_curve = self._response_curve_cache[n_grid]
        test_idx = self.get_test_indices(index)
        return t_grid, true_curve[test_idx]

    def __getitem__(self, index: int) -> DP_Dataset:
        train_idx, test_idx = self._split_indices(index)
        train_df = self.df.iloc[train_idx]
        test_df = self.df.iloc[test_idx]
        return DP_Dataset(
            X_train=self._df_numeric.iloc[train_idx].to_numpy(dtype=float),
            t_train=train_df["t"].to_numpy(dtype=float),
            y_train=train_df["y"].to_numpy(dtype=float),
            X_test=self._df_numeric.iloc[test_idx].to_numpy(dtype=float),
            t_optim=test_df["t_optim"].to_numpy(dtype=float),
            cepo_optim=test_df["cepo_optim"].to_numpy(dtype=float),
        )
