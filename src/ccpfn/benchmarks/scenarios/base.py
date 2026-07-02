from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd

class Scenario(ABC):
    """
    Base class for synthetic and semi-synthetic data construction.

    Data is constructed by loading covariates and applying dose_response to get 
    CEPOs at a given treatment level. Treatment assignment mechanisms are also 
    a part of this class and must be implemented in any subclass. 

    The outputs of the generation methods of this class are "raw", i.e. raw 
    pandas DataFrames or numpy ndarrays, not wrapped in any Dataset class (e.g. 
    CEPO_Dataset). That is to be done downstream.
    """

    # --- class-level constants each subclass overrides ---
    name: str
    t_min: float
    t_max: float
    y_noise_std: float
    low_outcome_is_better: bool

    @abstractmethod
    def load_covariates(self) -> pd.DataFrame: ...

    @abstractmethod
    def treatment(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """
        Treatment-generation function. 

        Args:
            df (pd.DataFrame): DataFrame of the covariates.
            rng (np.random.Generator): RNG for noise.
        """

    @abstractmethod
    def dose_response(self, df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
        """
        Noise-free dose-response function (CEPO).
        
        Args:
            df (pd.DataFrame): DataFrame of the covariates.
            t (np.ndarray): Array of treatment values.
        """
    
    # --- shared generation pipeline ---
    def _generate_observational(
        self, 
        rng: np.random.Generator
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generates a full observational dataset (X, t, y).
        """
        df = self.load_covariates()
        t = self.treatment(df, rng)
        y = self.dose_response(df, t) + rng.normal(0, self.y_noise_std, len(df))
        return df, t, y

    def generate_single_cepo(
        self, 
        rng: np.random.Generator
    ):
        """
        Generates a full single-CEPO dataset (X, t, y, t_test, true_cepo). Can be 
        used downstream in a CEPO_Dataset object.
        """
        df, t_obs, y_obs = self._generate_observational(rng)
        t_test = rng.uniform(self.t_min, self.t_max, len(df))
        true_cepo = self.dose_response(df, t_test)
        return df, t_obs, y_obs, t_test, true_cepo
    
    def generate_dosage_policy(
        self, 
        rng: np.random.Generator,
        n_grid: int
    ):
        """
        Generates a full optimal dosage policy dataset (X, t, y, t_optim, cepo_optim). Can 
        be used downstream in a DP_Dataset object.

        Args:
            n_grid (int): Number of grid points for the approximation of the dose-response
                          function on which to optimize.
        """
        df, t_obs, y_obs = self._generate_observational(rng)
        t_grid = np.linspace(self.t_min, self.t_max, n_grid)
        curve = np.stack(
            [self.dose_response(df, np.full(len(df), t)) for t in t_grid],
            axis=1,
        )
        if self.low_outcome_is_better:
            best_idx = curve.argmin(axis=1)
        else:
            best_idx = curve.argmax(axis=1)
        t_optim = t_grid[best_idx]
        cepo_optim = curve[np.arange(len(df)), best_idx]
        return df, t_obs, y_obs, t_optim, cepo_optim

    def generate_response_curve(
        self, 
        rng: np.random.Generator,
        n_grid: int,
    ):
        """
        Generates a full dose-response curve dataset
        """
        df, t_obs, y_obs = self._generate_observational(rng)
        t_grid = np.linspace(self.t_min, self.t_max, n_grid)
        true_curve = np.stack(
            [self.dose_response(df, np.full(len(df), t)) for t in t_grid],
            axis=1,
        )
        return df, t_obs, y_obs, t_grid, true_curve