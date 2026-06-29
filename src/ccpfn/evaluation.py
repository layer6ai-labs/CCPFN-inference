import numpy as np

def calculate_rmse(cepo_true: np.ndarray, cepo_pred: np.ndarray) -> float:
    """
    Calclulate the RMSE for CEPO estimation; nearly identical to PEHE.
    """
    return np.sqrt(np.mean((cepo_true - cepo_pred) ** 2))
