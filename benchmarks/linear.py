# Simple synthetic linear dataset based on the linear back-door dataset from
# https://arxiv.org/pdf/2210.08139

from typing import Any
import numpy as np

from .base import EvalDatasetCatalog, GeneralizedLinearDataset, CEPO_Dataset, DRC_Dataset

class SimpleLinearDataset(GeneralizedLinearDataset, EvalDatasetCatalog):
    def __init__(
        self, 
        n_tables: int, 
        test_ratio: float, 
        *args, 
        seed: int = 42,
        **kwargs
    ):
        """
        Synthetic dataset with a linear DRF.
        """
        GeneralizedLinearDataset.__init__(self, *args, **kwargs)
        EvalDatasetCatalog.__init__(self, n_tables=n_tables, name="simple-linear")

        self.n_tables = n_tables
        self.test_ratio = test_ratio
        self.seeds = [seed + i for i in range(self.n_tables)]

    def covariates2features(self, covariates):
        return covariates

    def __getitem__(self, index) -> tuple[CEPO_Dataset, DRC_Dataset]:
        if index >= self.n_tables:
            raise IndexError("Index out of range for the dataset catalog")
        
        np.random.seed(self.seeds[index])
        covariates, treatments, random_treatments, outcomes, true_cepos, true_drc_t_y = self.get_X_T_random_treatments_Y_cepo()

        indices = np.random.permutation(covariates.shape[0])
        split_idx = int(len(indices) * (1 - self.test_ratio))
        X_train, t_train, y_train = (
            covariates[indices[:split_idx]],
            treatments[indices[:split_idx]],
            outcomes[indices[:split_idx]]
        )
        X_test, t_test, cepo_test = (
            covariates[indices[split_idx:]], 
            random_treatments[indices[split_idx:]],
            true_cepos[indices[split_idx:]]
        )

        # Construct dataset objects
        cepo_dataset = CEPO_Dataset(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            X_test=X_test,
            t_test=t_test,
            true_cepo=cepo_test
        )
        drc_dataset = DRC_Dataset(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            true_drc_t_y=true_drc_t_y
        )

        return cepo_dataset, drc_dataset