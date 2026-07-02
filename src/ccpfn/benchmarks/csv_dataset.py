import itertools

import torch
import pandas as pd
from torch.utils.data import IterableDataset


class MultiCSVMetaDataset(IterableDataset):
    """
    Cycles through multiple CSV files in round-robin order.
    Each call to next() yields the full batch from the next CSV in sequence.
    """
    def __init__(self, csv_paths: list[str]):
        self.datasets = [SimpleCSVMetaDataset(p) for p in csv_paths]

    def __iter__(self):
        for ds in itertools.cycle(self.datasets):
            yield dict(
                X=ds.X,
                t=ds.t,
                y=ds.y,
                random_treatments=ds.random_treatments,
                mu_t=ds.mu_t,
            )


class SimpleCSVMetaDataset(IterableDataset):
    """
    Full-batch dataset from a single CSV with columns:
      - x_0..x_N       (covariates)
      - t, y           (observed treatment/outcome)
      - t_test, cepo_test  (query treatment/CEPO)

    Every row has all fields populated. calculate_loss handles the
    context/query split, just like normal training with
    ContinuousBackdoorDGPMetaDataset.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        x_cols = [c for c in df.columns if c.startswith('x_')]

        N = len(df)
        self.X = torch.tensor(df[x_cols].values, dtype=torch.float32).unsqueeze(0)           # (1, N, D)
        self.t = torch.tensor(df['t'].values, dtype=torch.float32).unsqueeze(0)               # (1, N)
        self.y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(0)               # (1, N)
        self.random_treatments = torch.tensor(df['t_test'].values, dtype=torch.float32).unsqueeze(0)  # (1, N)
        self.mu_t = torch.tensor(df['cepo_test'].values, dtype=torch.float32).unsqueeze(0)    # (1, N)

        print(f"Loaded full-batch dataset: {N} rows, {len(x_cols)} features")

    def __iter__(self):
        while True:
            yield dict(
                X=self.X,
                t=self.t,
                y=self.y,
                random_treatments=self.random_treatments,
                mu_t=self.mu_t,
            )
