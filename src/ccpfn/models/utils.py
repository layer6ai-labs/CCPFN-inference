import torch

# Covariate types
OBSERVED_BACKDOOR = 100.0
DUMMY = 0.0


def pad_x_and_t(X: torch.Tensor, num_features: int = 100, pad_value: float = DUMMY):
    """Zero-pads features (treatment and covariates) 

    Args:
        X (torch.Tensor): data to pad, treatments and covariates (concatenated)
        num_features (int): number of features total (treatment_dim + num_covariates)
        pad_value (float): data to pad with
    """
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.ones((*X.shape[:-1], num_features - n_features), device=X.device) * pad_value
    return torch.cat([X, zero_feature_padding], dim=-1)