from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List

import numpy as np

SamplerType = Callable[[tuple], np.ndarray]


@dataclass
class CEPO_Dataset:  # conditional expected potential outcomes
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    t_test: np.ndarray
    true_cepo: np.ndarray  # y_test


@dataclass
class DRC_Dataset:  # dose-response curves
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    true_drc_t_y: list[tuple[float]]  # list of (t, y) pairs on dose-reponse curve


@dataclass
class DP_Dataset:  # dosage policy
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    t_optim: np.ndarray  # per-row optimal treatment
    cepo_optim: np.ndarray  # dose-response evaluated at t_optim


### Helper functions ###
def admit_outcome_function(t: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray, x5: np.ndarray, x6: np.ndarray) -> np.ndarray:
    """The outcome function used in the ADMIT paper:
    "Generalizing Bounds for Estimating Causal Effects of Continuous Treatments"
    """
    if t.ndim == 1:
        return np.cos(2 * np.pi * (t - 0.5)) * (t ** 2 + 4 * np.sin(x4) * (np.maximum(x1, x6) ** 3) / (1 + 2 * x3 ** 2))
    else:
        return np.cos(2 * np.pi * (t - 0.5)) * (t ** 2 + 4 * (np.sin(x4) * (np.maximum(x1, x6) ** 3)).reshape(-1, 1) / (1 + 2 * x3 ** 2).reshape(-1, 1))


### Dataset Catalogs ###
class EvalDatasetCatalog(ABC):
    """
    The dataset catalog is a collection of datasets used for evaluating the model.
    """

    def __init__(self, n_tables: int, name: str):
        self.n_tables = n_tables
        self.name = name

    def __len__(self):
        return self.n_tables

    def __str__(self):
        return self.name

    @abstractmethod
    def __getitem__(self, index) -> Any:
        raise NotImplementedError("This method should be implemented by the subclass")


### Samplers ###
class LaplaceSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        if shape is None:
            return np.random.laplace(self.loc, self.scale)
        else:
            return np.random.laplace(self.loc, self.scale, shape)


class UniformSampler:
    """Samples noise from a uniform distribution upon calling"""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        if shape is None:
            ret = np.random.rand()
        else:
            ret = np.random.rand(*shape)
        return ret * (self.high - self.low) + self.low


class GaussianSampler:
    """Samples noise from a gaussian distribution upon calling"""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        if shape is None:
            return np.random.normal(self.loc, self.scale)
        else:
            return np.random.normal(self.loc, self.scale, shape)


class UniformIntegerSampler:
    """Samples noise from a uniform integer distribution upon calling"""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __call__(self, shape: tuple | None = None) -> np.ndarray | int:
        if shape is None:
            ret = np.random.randint(self.low, self.high + 1)
        else:
            ret = np.random.randint(self.low, self.high + 1, shape)
        return ret


### DGPs ###
class GeneralizedLinearDataset(ABC):
    """
    Modelling a synthetic DGP where E[T | X] can be expressed as a relatively simple function of 
    X, and E[Y_t | X] can be expressed as a relatively simple function of X and t.
    """
    def __init__(
        self,
        n_samples: int = 4096,
        n_random_treatments: int = 10,
        x_dim_dist: Callable[[], int] = UniformIntegerSampler(4, 25),
        noise_samplers: List[SamplerType] | SamplerType = [
            GaussianSampler(0.0, 0.1),
            UniformSampler(-0.1, 0.1),
            LaplaceSampler(0, 0.1),
        ],
        weight_sampler: List[SamplerType] | SamplerType = UniformSampler(-0.5, 0.5),
        covariate_sampler: List[SamplerType] | SamplerType = UniformSampler(-1.0, 1.0),
        standardize_treatment: bool = True,
        num_subintvls: int = 40,
    ) -> None:
        """
        Args
            n_samples: Number of samples to generate.
            x_dim_dist: A callable that returns the number of dimensions of covariates.
            noise_samplers: A list of callable samplers for exogenous noise.
            weight_sampler:
                A list of callable samplers for the linear weights that will be used
                for the linear transformation of the features induced by the covariates
                (not the covariates directly)
            covariate_sampler:
                A list of callable samplers for the covariates.
            standardize_treatment: Whether to standardize the treatment logits.
            num_subintvls: Number of subintervals to use to approximate DRC.
        """
        if isinstance(noise_samplers, list):
            self.noise_samplers = noise_samplers
        else:
            self.noise_samplers = [noise_samplers]

        # for linear weights
        if isinstance(weight_sampler, list):
            self.weight_sampler = weight_sampler
        else:
            self.weight_sampler = [weight_sampler]

        # for the covariates
        if isinstance(covariate_sampler, list):
            self.covariate_sampler = covariate_sampler
        else:
            self.covariate_sampler = [covariate_sampler]

        self.x_dim_dist = x_dim_dist
        self.n_samples = n_samples
        self.n_random_treatments = n_random_treatments
        self.standardize_treatment = standardize_treatment
        self.num_subintvls = num_subintvls

    def sample_exogenous_noise(self, shape) -> np.ndarray:
        chosen_exogenous_sampler = self.noise_samplers[np.random.randint(len(self.noise_samplers))]
        return chosen_exogenous_sampler(shape)

    def sample_weights(self, shape) -> np.ndarray:
        chosen_weight_sampler = self.weight_sampler[np.random.randint(len(self.weight_sampler))]
        return chosen_weight_sampler(shape)

    def sample_covariates(self, shape) -> np.ndarray:
        chosen_covariate_sampler = self.covariate_sampler[np.random.randint(len(self.covariate_sampler))]
        return chosen_covariate_sampler(shape)
    
    def sample_random_treatments(self, shape) -> np.ndarray:
        random_treatments = np.random.rand(*shape)
        return random_treatments

    @abstractmethod
    def covariates2features(self, covariates):
        pass

    def get_X_T_random_treatments_Y_cepo(self, seed: int | None = None):
        if seed:
            np.random.seed(seed)

        n_dims = self.x_dim_dist()
        covariates = self.sample_covariates((self.n_samples, n_dims))

        covariate_features = self.covariates2features(covariates)
        features_dims = covariate_features.shape[1]

        # Treatment assignment
        w_t = self.sample_weights((features_dims,))
        treatment_pre_logits = np.einsum("np,p->n", covariate_features, w_t)
        treatment_logits = treatment_pre_logits + self.sample_exogenous_noise((self.n_samples,))
        if self.standardize_treatment:
            treatment_logits = (treatment_logits - treatment_logits.mean()) / (treatment_logits.std() + 1e-20)
        t = 1 / (1 + np.exp(-treatment_logits))  # always maps treatments to [0, 1]

        # Observed outcomes
        w_y_x = self.sample_weights((features_dims,))
        w_y_t = self.sample_weights(None)  # scalar
        expected_y = np.einsum("np,p->n", covariate_features, w_y_x) + w_y_t * t
        y = expected_y + self.sample_exogenous_noise((self.n_samples,))

        # Generate random treatments
        random_treatments = self.sample_random_treatments((self.n_samples, self.n_random_treatments))

        # Calculate CEPOs
        covariates_part = np.einsum("np,p->n", covariate_features, w_y_x).reshape(-1, 1)
        mu_t = covariates_part + w_y_t * random_treatments

        # Construct array of DRC values
        t_test = np.linspace(0.0, 1.0, self.num_subintvls)
        drc_t_y_pairs = []
        for drc_t in t_test: 
            drc_y_before_mean = covariates_part + w_y_t * drc_t 
            drc_y = drc_y_before_mean.mean()
            drc_t_y_pairs.append((drc_t, drc_y))

        return covariates, t, random_treatments, y, mu_t, drc_t_y_pairs


class LegacyLinearDataset(ABC):
    """
    Modelling a synthetic DGP where P(Y0 | X), P(Y1 | X), and logit(P(T | X)) can
    be expressed as a linear function of a feature representation of X followed by
    non-linear transformation.

    This is a legacy class for the polynomial and sinusoidal benchmarks, modified
    from the binary treatment case.
    """

    def __init__(
        self,
        n_samples: int = 2048,
        n_random_treatments: int = 10,
        x_dim_dist: Callable[[], int] = UniformIntegerSampler(5, 10),
        noise_samplers: List[SamplerType] | SamplerType = [
            GaussianSampler(0.0, 1.0),
            UniformSampler(-1.0, 1.0),
            LaplaceSampler(0, 1.0),
        ],
        weight_sampler: List[SamplerType] | SamplerType = UniformSampler(-5.0, 5.0),
        covariate_sampler: List[SamplerType] | SamplerType = UniformSampler(-2.0, 2.0),
        standardize_treatment: bool = True,
        num_subintvls: int = 40,
    ) -> None:
        """
        Args
            n_samples: Number of samples to generate.
            x_dim_dist: A callable that returns the number of dimensions of covariates.
            noise_samplers: A list of callable samplers for exogenous noise.
            weight_sampler:
                A list of callable samplers for the linear weights that will be used
                for the linear transformation of the features induced by the covariates
                (not the covariates directly)
            covariate_sampler:
                A list of callable samplers for the covariates.
            standardize_treatment: Whether to standardize the treatment logits.
            num_subintvls: Number of subintervals to use to approximate DRC.
        """
        self.x_dim_dist = x_dim_dist

        if isinstance(noise_samplers, list):
            self.noise_samplers = noise_samplers
        else:
            self.noise_samplers = [noise_samplers]

        # for linear weights
        if isinstance(weight_sampler, list):
            self.weight_sampler = weight_sampler
        else:
            self.weight_sampler = [weight_sampler]

        # for the covariates
        if isinstance(covariate_sampler, list):
            self.covariate_sampler = covariate_sampler
        else:
            self.covariate_sampler = [covariate_sampler]

        self.n_samples = n_samples
        self.standardize_treatment = standardize_treatment
        self.standardize_outcome = standardize_outcome

    def sample_exogenous_noise(self, shape) -> np.ndarray:
        chosen_exogenous_sampler = self.noise_samplers[np.random.randint(len(self.noise_samplers))]
        return chosen_exogenous_sampler(shape)

    def sample_weights(self, shape) -> np.ndarray:
        chosen_weight_sampler = self.weight_sampler[np.random.randint(len(self.weight_sampler))]
        return chosen_weight_sampler(shape)

    def sample_covariates(self, shape) -> np.ndarray:
        chosen_covariate_sampler = self.covariate_sampler[np.random.randint(len(self.covariate_sampler))]
        return chosen_covariate_sampler(shape)

    def sample_random_treatments(self, shape) -> np.ndarray:
        random_treatments = np.random.rand(*shape)
        return random_treatments

    @abstractmethod
    def covariates2features(self, covariates):
        pass

    @abstractmethod
    def post_nonlinear(self, random_variable):
        pass

    def get_X_T_propensities_Y0_Y1_E_Y0_E_Y1_outcomes(self, seed: int | None = None):
        if seed:
            np.random.seed(seed)

        n_dims = self.x_dim_dist()
        covariates = self.sample_covariates((self.n_samples, n_dims))

        covariate_features = self.covariates2features(covariates)
        features_dims = covariate_features.shape[1]

        # treatment assignment
        w_T = self.sample_weights((features_dims,))
        treatment_pre_logits = np.einsum("np,p->n", covariate_features, w_T)
        treatment_logits = self.post_nonlinear(treatment_pre_logits) + self.sample_exogenous_noise((self.n_samples,))
        if self.standardize_treatment:
            treatment_logits = (treatment_logits - treatment_logits.mean()) / (treatment_logits.std() + 1e-20)
        treatments = 1 / (1 + np.exp(-treatment_logits))

        # observed outcomes
        w_Y = self.sample_weights((features_dims,))
        expected_y = self.post_nonlinear(np.einsum("np,p->n", covariate_features, w_Y))
        y = expected_y + self.sample_exogenous_noise((self.n_samples,))

        # generate random treatments and their CEPOs
        random_treatments = self.sample_random_treatments((self.n_samples, self.n_random_treatments))

        outcomes = np.where(treatments == 1, y1, y0)
        if self.standardize_outcome:
            outcomes_mean, outcomes_std = outcomes.mean(), outcomes.std() + 1e-20
        else:
            outcomes_mean, outcomes_std = 0, 1
        outcomes = (outcomes - outcomes_mean) / outcomes_std
        y0, y1 = (y0 - outcomes_mean) / outcomes_std, (y1 - outcomes_mean) / outcomes_std
        E_y0, E_y1 = (E_y0 - outcomes_mean) / outcomes_std, (E_y1 - outcomes_mean) / outcomes_std
        return covariates, treatments, treatment_probs, y0, y1, E_y0, E_y1, outcomes
        raise NotImplementedError("Need to finish implementing LegacyLinearDataset")


class GeneralizedADMITDataset(ABC):
    def __init__(
        self, 
        n_samples: int = 4096, 
        n_features: int = 6,
        n_random_treatments: int = 10, 
        covariate_sampler: SamplerType = UniformSampler(0.0, 1.0),
        treatment_noise_sampler: SamplerType = GaussianSampler(0.0, 0.5),
        outcome_noise_sampler: SamplerType = GaussianSampler(0.0, 0.5),
        num_subintvls: int = 40,
    ) -> None:
        """Synthetic dataset based on that generated in the paper "Generalizing Bounds
        for Estimating Causal Effects of Continuous Treatments". 

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features to generate. Paper set n_features = 6.
            n_random_treatments: Number of random treatments to generate for t_query.
            covariate_sampler: Samplers for the covariates.
            treatment_sampler: Samplers for the treatment variable.
            outcome_samplers: Samplers for the outcome variable.
            num_subintvls: Number of subintervals to use to approximate DRC.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_random_treatments = n_random_treatments
        self.covariate_sampler = [covariate_sampler]
        self.treatment_noise_sampler = [treatment_noise_sampler]
        self.outcome_noise_sampler = [outcome_noise_sampler]
        self.num_subintvls = num_subintvls

    def sample_covariates(self, shape) -> np.ndarray:
        chosen_covariate_sampler = self.covariate_sampler[np.random.randint(len(self.covariate_sampler))]
        return chosen_covariate_sampler(shape)

    def sample_observed_treatment_noise(self, shape) -> np.ndarray:
        chosen_treatment_noise_sampler = self.treatment_noise_sampler[np.random.randint(len(self.treatment_noise_sampler))]
        return chosen_treatment_noise_sampler(shape)

    def sample_random_treatments(self, shape) -> np.ndarray:
        """Generates random treatments uniformly in [0, 1]"""
        random_treatments = np.random.rand(*shape)
        return random_treatments
    
    def sample_observed_outcome_noise(self, shape) -> np.ndarray:
        chosen_outcome_noise_sampler = self.outcome_noise_sampler[np.random.randint(len(self.outcome_noise_sampler))]
        return  chosen_outcome_noise_sampler(shape)
    
    def get_X_T_random_treatments_Y_cepo(self, seed: int | None = None):
        if seed:
            np.random.seed(seed)
        
        # Generate covariates
        covariates = self.sample_covariates((self.n_samples, self.n_features))
        x1 = covariates[:, 0]
        x2 = covariates[:, 1]
        x3 = covariates[:, 2]
        x4 = covariates[:, 3]
        x5 = covariates[:, 4]
        x6 = covariates[:, 5]

        # Assign observed treatment
        t_logit_mean = (10. * np.sin(np.maximum(x1, np.maximum(x2, x3))) + np.maximum(x3, np.maximum(x4, x5)) ** 3)/(1. + (x1 + x5) ** 2) + np.sin(0.5 * x3) * (1. + np.exp(x4 - 0.5 * x3)) + x3 ** 2 + 2. * np.sin(x4) + 2. * x5 - 6.5
        t_logit = t_logit_mean + self.sample_observed_treatment_noise(self.n_samples)
        t = 1.0 / (1 + np.exp(-t_logit))

        # Sample random treatments (for CEPO evaluation)
        random_treatments = self.sample_random_treatments((self.n_samples, self.n_random_treatments))

        # Assign observed outcomes
        y_mean_observed = admit_outcome_function(t, x1, x2, x3, x4, x5, x6)
        y_observed = y_mean_observed + self.sample_observed_outcome_noise(self.n_samples)

        # Assign CEPOs corresponding to random_treatments
        mu_t = admit_outcome_function(random_treatments, x1, x2, x3, x4, x5, x6)

        # Construct array of DRC values
        t_test = np.linspace(0.0, 1.0, self.num_subintvls)
        drc_t_y_pairs = []
        for drc_t in t_test: 
            drc_y_before_mean = admit_outcome_function(drc_t, x1, x2, x3, x4, x5, x6)
            drc_y = drc_y_before_mean.mean()
            drc_t_y_pairs.append((drc_t, drc_y))

        return covariates, t, random_treatments, y_observed, mu_t, drc_t_y_pairs