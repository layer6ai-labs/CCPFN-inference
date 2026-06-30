import numpy as np
import torch
import torch.nn as nn

from .transformer_layer import TransformerEncoderLayer


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)

def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5

def normalize_data(data, eval_pos):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-6
    data = (data - mean) / std
    return data

def clip_outliers(data, eval_pos, n_sigma=4):
    assert len(data.shape) == 3, "X must be T,B,H"
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)

def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


class TabDPTLongContextModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_covariates: int,
        treatment_dim: int,
        nbins: int,
    ):
        """TabDPTLongContextModel initialization.

        Args:
            dropout (float): Dropout rate
            n_out (int): Number of output classes (legacy from non-causal TabDPT model)
            nhead (int): Number of attention heads
            nhid (int): Hidden dimension
            ninp (int): Input dimension
            nlayers (int): Number of transformer layers
            num_covariates (int): Number of covariates
            treatment_dim (int): Dimension of treatment representation
            nbins (int): Number of bins to quantize range to (for regression-as-classification)
        """
        super().__init__()
        self.n_out = n_out
        self.ninp = ninp
        self.nbins = nbins
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_covariates = num_covariates
        self.treatment_dim = treatment_dim
        self.x_and_t_encoder = nn.Linear(num_covariates + treatment_dim, ninp, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.t_encoder = nn.Sequential(
            nn.Linear(treatment_dim, nhid, bias=True), nn.GELU(), nn.Linear(nhid, ninp, bias=True), nn.LayerNorm(ninp)
        )
        self.y_encoder = nn.Linear(1, ninp, bias=True)
        self.head = nn.Sequential(
            nn.Linear(ninp, nhid, bias=False), nn.GELU(), nn.Linear(nhid, n_out + nbins, bias=False)
        )
        self.xnorm = nn.LayerNorm(ninp, bias=False)
        self.tnorm = nn.LayerNorm(ninp, bias=False)
        self.ynorm = nn.LayerNorm(ninp, bias=False)

        # For training: zero-init second layer of t_encoder so residual branch is identity 
        # at start (t_encoder(t_src) = 0)
        nn.init.zeros_(self.t_encoder[2].weight)

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        return_log_act_norms: bool = False,
    ) -> torch.Tensor:
        """Forward pass of TabDPTLongContextModel.

        Args:
            x_src (torch.Tensor): Input treatment and covariates; treatment is first value in last dim
            y_src (torch.Tensor): Target values
            return_log_act_norms (bool): Whether to return activation norms for logging

        Returns:
            torch.Tensor: Predicted distribution of target values (logits)
        """
        context_length = y_src.shape[0]
        B = y_src.shape[1]

        # Extract treatment and covariates before preprocessing
        t_src = x_src[:, :, 0:self.treatment_dim]
        x_src = x_src[:, :, self.treatment_dim:]

        # Treatment: normalize and clip outliers
        if self.treatment_dim == 1:
            t_src = normalize_data(t_src, context_length)
            t_src = clip_outliers(t_src, context_length, n_sigma=10)
            t_src = torch.nan_to_num(t_src, nan=0)

        # Covariates: normalize and clip outliers
        x_src = normalize_data(x_src, context_length)
        x_src = clip_outliers(x_src, context_length, n_sigma=10)
        x_src = torch.nan_to_num(x_src, nan=0)

        # Concat t and x for linear encoder
        x_and_t_src = torch.cat([t_src, x_src], -1)

        # Encode and normalize
        x_and_t_src = self.xnorm(self.x_and_t_encoder(x_and_t_src))
        t_src = self.tnorm(self.t_encoder(t_src))
        y_src = self.ynorm(self.y_encoder(y_src.unsqueeze(-1)))
        
        train_x_and_t = x_and_t_src[:context_length] + t_src[:context_length] + y_src
        src = torch.cat([train_x_and_t, x_and_t_src[context_length:] + t_src[context_length:]], 0)

        # Transformer layers
        for l, layer in enumerate(self.transformer_encoder):
            src = layer(src, context_length)

        # Final head
        pred = self.head(src)

        # Random hack that works - not in current TabDPT
        pred = 30 * torch.tanh(pred / (7.5 * src.size(-1) ** 0.5))

        if return_log_act_norms:
            return pred[context_length:], log_act_norms
        else:
            return pred[context_length:]


    @classmethod
    def load(cls, model_state, config):
        """Loads weights for a TabDPTLongContextModel. Supports partial weight instantiation. 
        """

        assert config.model.max_num_classes > 2

        model = TabDPTLongContextModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_covariates=config.model.max_num_covariates,
            treatment_dim=config.model.treatment_dim,
            nbins=config.model.nbins,
        )

        module_prefix = "_orig_mod."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        if "encoder.weight" in model_state and "x_and_t_encoder.weight" not in model_state:
            old_weight = model_state.pop("encoder.weight")  # (ninp, num_covariates)
            old_bias = model_state.pop("encoder.bias")      # (ninp,)

            new_weight = torch.zeros_like(model.x_and_t_encoder.weight)  # (ninp, treatment_dim + num_covariates)
            new_weight[:, model.treatment_dim:] = old_weight  # covariate columns
            # treatment columns stay zero

            model_state["x_and_t_encoder.weight"] = new_weight
            model_state["x_and_t_encoder.bias"] = old_bias

        # Determine which transformer layers exist in the checkpoint
        ckpt_layer_ids = {
            int(k.split(".")[1]) for k in model_state if k.startswith("transformer_encoder.")
        }
        ckpt_max_layer = max(ckpt_layer_ids) if ckpt_layer_ids else -1

        expected_new = {"t_encoder", "tnorm"}
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        unexpected_missing = [
            k for k in missing
            if not any(k.startswith(p) for p in expected_new)
            and not (
                k.startswith("transformer_encoder.")
                and int(k.split(".")[1]) > ckpt_max_layer
            )
        ]
        if unexpected_missing:
            raise ValueError(f"Unexpectedly missing weights: {unexpected_missing}")

        model.eval()
        return model