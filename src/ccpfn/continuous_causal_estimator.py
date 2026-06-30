import math
import os
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

from .models import ContinuousInContextModel

def is_hf_model_path(model_path: str) -> bool:
    """
    Check if a given path is a Hugging Face model path (repo_id/model_name).

    Args:
        model_path: The model path to check

    Returns:
        bool: True if it's a Hugging Face model path, False otherwise
    """
    # Check if path doesn't exist locally but follows the pattern of org/repo or user/repo
    if not os.path.exists(model_path) and "/" in model_path and model_path.count("/") == 1:
        return True
    return False

def download_from_hf_hub(model_path: str, cache_dir: str) -> str:
    """
    Download a model from the Hugging Face Hub.

    Args:
        model_path: The model path in format 'org/repo' or 'user/repo'
        cache_dir: Optional directory to cache the downloaded model

    Returns:
        str: Path to the downloaded model file
    """
    # Download the model
    local_path = hf_hub_download(
        repo_id=model_path,
        filename="causalpfn_v0.pt",
        cache_dir=cache_dir,
    )
    return local_path


class ContinuousCausalEstimator(ABC):
    def __init__(
        self, 
        device: str, 
        model_path: str = "chris-L6/CCPFN",
        max_context_length: int = 4096,
        max_query_length: int = 4096,
        verbose: bool = False, 
        cache_dir: str | None = None,
        icl_model: ContinuousInContextModel | None = None,
    ):
        self.model_path = model_path
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(Path.home(), ".cache", "ccpfn")
        self.icl_model: ContinuousInContextModel = icl_model

        self.device = device
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length

        # The maximum number of features to use for the model. If the number of features are
        # larger than this value, the model will apply PCA to reduce the dimensionality.
        self.max_feature_size = None
        self.x_dim_transformer = FunctionTransformer()  # identity transformer by default

        self.n_folds = n_folds

        self.X_train, self.t_train, self.y_train = None, None, None
        self.temperature = 1.0
        self.prediction_temperature = 1.0

        self.verbose = verbose

    def _check_fitted(self):
        if self.X_train is None or self.t_train is None or self.y_train is None or self.icl_model is None:
            raise ValueError("The estimator must be fitted before calling the estimate function.")

    def load_model(self):
        """
        Load the model from the specified path or download it from Hugging Face.
        """
        if self.model_path is not None:
            model_path = self.model_path

            # Check if the model path is a Hugging Face model path
            if is_hf_model_path(model_path):
                model_path = download_from_hf_hub(model_path, self.cache_dir)

            # Load the model from the local path
            ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
            model_state = ckpt["model_state_dict"]
            config = ckpt["model_config"]

            self.icl_model = ContinuousInContextModel.load(model_state=model_state, model_config=config).to(self.device)
        elif self.icl_model is not None:
            # If icl_model is provided, use it directly
            self.icl_model.to(self.device)
            config = self.icl_model.model_config
        else:
            raise ValueError("Either model_path or icl_model must be provided.")

        if config["model_type"] == "tabdpt":
            self.max_feature_size = config["model"]["max_num_covariates"]
            self.x_dim_transformer = TruncatedSVD(n_components=self.max_feature_size, algorithm="arpack")

    @torch.no_grad()
    def _predict_cepo(
        self,
        X_context: np.ndarray,
        t_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        t_query: np.ndarray,
        temperature: float,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if self.icl_model is None:
            raise ValueError("CausalEstimator must be fitted before calling _predict_cepo.")

        temperature = torch.tensor([temperature], device=self.device)
        self.icl_model: ContinuousInContextModel
        self.icl_model.eval()

        # list all of the point estimates as well as distributional estimates
        # of the CEPO in all_cepo and all_samples, respectively
        all_cepo = np.zeros((X_query.shape[0],), dtype=X_query.dtype)
        if n_samples is not None:
            all_samples = np.zeros((X_query.shape[0], n_samples), dtype=X_query.dtype)
        
        if X_context.shape[0] > self.max_context_length:
            if seed is not None:
                np.random.seed(seed)
            idx_c = np.random.choice(X_context.shape[0], self.max_context_length)
            x_c = X_context[idx_c]
            t_c = t_context[idx_c]
            y_c = y_context[idx_c]
        else:
            x_c = X_context
            t_c = t_context
            y_c = y_context

        # If the query is large, we split it into batches
        pbar = tqdm(range(X_query.shape[0]), desc="Predicting CEPO", total=X_query.shape[0], disable=not self.verbose)
        start_idx = 0
        while start_idx < X_query.shape[0]:
            end_idx = min(start_idx + self.max_query_length, X_query.shape[0])
            x_q = X_query[start_idx:end_idx]
            t_q = t_query[start_idx:end_idx]
            res = self.icl_model.predict_cepo(
                # shape: (1, context_size, num_features)
                X_context=torch.from_numpy(x_c).to(self.device).unsqueeze(0).float(),
                # shape: (1, context_size)
                t_context=torch.from_numpy(t_c).to(self.device).unsqueeze(0).float(),
                y_context=torch.from_numpy(y_c).to(self.device).unsqueeze(0).float(),
                # shape: (1, query_size, num_features)
                X_query=torch.from_numpy(x_q).to(self.device).unsqueeze(0).float(),
                # shape: (1, query_size)
                t_query=torch.from_numpy(t_q).to(self.device).unsqueeze(0).float(),
                n_samples=n_samples,
                temperature=temperature,
            )
            if n_samples is None:
                cepo = res.squeeze(0).squeeze(0)  # shape: (query_size,)
            else:
                cepo, samples = (
                    res[0].squeeze(0).squeeze(0),
                    res[1].squeeze(0).squeeze(0),
                )  # shapes: (query_size,), (query_size, n_samples)
                all_samples[start_idx:end_idx] = samples.cpu().numpy()
            all_cepo[start_idx:end_idx] = cepo.cpu().numpy()            
            pbar.update(end_idx - start_idx)
            start_idx = end_idx
        pbar.close()
        
        if n_samples is not None:
            return all_cepo, all_samples

        return all_cepo

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "ContinuousCausalEstimator":
        """
        Fit the model using the provided data.

        Args:
            X (np.ndarray): The observational covariate data with shape [N, D].
            t (np.ndarray): The observational treatment data with shape [N].
            y (np.ndarray): The observational outcome data with shape [N].
        """
        self.temperature = 1.0

        # load the model
        self.load_model()

        # set the x_dim_transform and transform the data
        if self.max_feature_size is not None and X.shape[1] > self.max_feature_size:
            X = self.x_dim_transformer.fit_transform(X)

        self.X_train = X
        self.t_train = t
        self.y_train = y

        return self

class CEPOEstimator(ContinuousCausalEstimator):
    def estimate_cepo(
        self, 
        X: np.ndarray, 
        t: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the conditional expected potential outcome (CEPO) using the fitted model.

        Args:
            X (np.ndarray): The input data (covariates) with shape [N', D]
            t (np.ndarray): The input data (treatment values) with shape [N']

        Returns: 
            all_cepo: The 1-D array of all expected potential outcomes mu_t(X), where X = 
                      input covariates, t = input treatments.
        """
        self._check_fitted()

        X_context = self.X_train
        t_context = self.t_train
        y_context = self.y_train
        X_query = X
        if self.max_feature_size is not None and X_query.shape[1] > self.max_feature_size:
            X_query = self.x_dim_transformer.transform(X_query)

        t_query = t

        all_cepo = self._predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=X_query,
            t_query=t_query,
            temperature=self.prediction_temperature
        )

        return all_cepo

class Prescriber(ContinuousCausalEstimator):
    def estimate_optimal_treatment(
        self,
        X: np.ndarray,
        n_grid: int,
        t_min: float,
        t_max: float,
        lower_is_better: bool,
    ) -> np.ndarray:
        """
        For each row of X, evaluate the model's CEPO on a uniform grid of
        n_grid treatments in [t_min, t_max] and return the argmin/argmax.

        Args:
            X (np.ndarray): Covariates with shape [N, D].
            n_grid (int): Number of grid points in [t_min, t_max].
            t_min (float): Lower bound of treatment grid.
            t_max (float): Upper bound of treatment grid.
            lower_is_better (bool): If True, choose the treatment minimizing CEPO,
                otherwise the maximizing one.

        Returns:
            t_hat (np.ndarray): Per-row optimal treatment, shape [N].
        """
        self._check_fitted()
        if n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {n_grid}")

        t_grid = np.linspace(t_min, t_max, n_grid)
        n = X.shape[0]
        if self.max_feature_size is not None and X.shape[1] > self.max_feature_size:
            X = self.x_dim_transformer.transform(X)
        X_query = np.repeat(X, n_grid, axis=0)
        t_query = np.tile(t_grid, n)

        all_cepo = self._predict_cepo(
            X_context=self.X_train,
            t_context=self.t_train,
            y_context=self.y_train,
            X_query=X_query,
            t_query=t_query,
            temperature=self.prediction_temperature,
        ).reshape(n, n_grid)

        best = all_cepo.argmin(axis=1) if lower_is_better else all_cepo.argmax(axis=1)
        return t_grid[best]