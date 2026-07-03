# A synthetic dataset based on the simulated dataset generated in 
# https://github.com/CausalTeam/ADMIT/blob/main/data/dataset.py and 
# https://github.com/lushleaf/varying-coefficient-net-with-functional-tr/blob/main/data/simu1.py

import numpy as np
import pandas as pd

from .base import Scenario

class ADMIT(Scenario):
    """Fully-synthetic scenario adapted from 
    https://github.com/CausalTeam/ADMIT/blob/main/data/dataset.py and 
    https://github.com/lushleaf/varying-coefficient-net-with-functional-tr/blob/main/data/simu1.py. 
    """
    name = "ADMIT"
    t_min = 0.0
    t_max = 1.0
    y_noise_std = 0.5
    treatment_noise_std = 0.5
    low_outcome_is_better = False

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 10_000,
    ):
        self.seed = seed
        self.n_samples = n_samples
        self._df_cache: pd.DataFrame | None = None

    def _generate_raw(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        covariates = rng.uniform(0.0, 1.0, size=(self.n_samples, 6))
        df = pd.DataFrame(
            covariates,
            columns=[f"x_{i}" for i in range(6)],
        )
        return df.astype(float)

    def load_covariates(self) -> pd.DataFrame:
        if self._df_cache is None:
            self._df_cache = self._generate_raw()
        return self._df_cache

    def treatment(
        self, 
        df: pd.DataFrame, 
        rng: np.random.Generator,
    ) -> np.ndarray:
        x1, x2, x3, x4, x5, x6 = [df[f"x_{i}"] for i in range(6)]
        t_logit_mean = (10. * np.sin(np.maximum(x1, np.maximum(x2, x3))) + np.maximum(x3, np.maximum(x4, x5)) ** 3)/(1. + (x1 + x5) ** 2) + np.sin(0.5 * x3) * (1. + np.exp(x4 - 0.5 * x3)) + x3 ** 2 + 2. * np.sin(x4) + 2. * x5 - 6.5
        noise = rng.normal(0.0, self.treatment_noise_std, size=len(t_logit_mean))
        t_logit = t_logit_mean + noise
        t = 1.0 / (1 + np.exp(-t_logit))

        return t
    
    def dose_response(self, df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
        x1, x2, x3, x4, x5, x6 = [df[f"x_{i}"] for i in range(6)]
        if t.ndim == 1:
            return np.cos(2 * np.pi * (t - 0.5)) * (t ** 2 + 4 * np.sin(x4) * (np.maximum(x1, x6) ** 3) / (1 + 2 * x3 ** 2))
        else:
            return np.cos(2 * np.pi * (t - 0.5)) * (t ** 2 + 4 * (np.sin(x4) * (np.maximum(x1, x6) ** 3)).reshape(-1, 1) / (1 + 2 * x3 ** 2).reshape(-1, 1))
