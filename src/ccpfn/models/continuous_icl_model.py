import math 
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TabDPTLongContextModel
from .utils import pad_x_and_t


class ContinuousInContextModel(nn.Module):
    def __init__(
        self,
        model: TabDPTLongContextModel,
        model_config: dict,  # the config containing the constructor arguments for the model
        sigma: float = 0.5,  # overriden in conf/train.yaml
        vmin: float = -10.0,
        vmax: float = 10.0,
    ):
        super().__init__()
        self.model: nn.Module = model
        self.model_config = model_config

        # self.prepare_input is to be called for each of the models to do any model-specific preprocessing
        self.prepare_input = lambda x, y: (pad_x_and_t(x, model.num_covariates + model.treatment_dim), y)
        self.nbins = model_config["model"]["nbins"]
        model_config["model_type"] = "tabdpt"
        model_config["sigma"] = sigma
        self.sigma = sigma

        # NOTE: These variables are stored to avoid re-initializing them for each forward pass
        self.vmin = vmin
        self.vmax = vmax

        bin_edges = torch.linspace(self.vmin, self.vmax, self.nbins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + 0.5 * bin_width  # shape: (nbins,)

        self.register_buffer("bin_edges", bin_edges)  # (nbins+1,)
        self.register_buffer("bin_width", bin_width)  # () – 0-D tensor
        self.register_buffer("bin_centers", bin_centers)  # (nbins,)

    def _predict_mean(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_centers, dim=-1)
    
    def _predict_mode(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        mode_indices = torch.argmax(probs, dim=-1)
        return self.bin_centers[mode_indices]

    def _sample_from_logits(self, logits: torch.Tensor, n_samples: int) -> torch.Tensor:
        # Flatten trailing dim to (..., nbins)
        orig_shape = logits.shape[:-1]

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # shape: (..., nbins)

        # Reshape for sampling: (batch_size, nbins)
        logits_reshaped = probs.reshape(-1, self.nbins)

        # Sample indices
        sampled_indices = torch.multinomial(
            logits_reshaped, num_samples=n_samples, replacement=True
        )  # (batch_size, n_samples)

        # Convert indices to values using bin_centers
        samples = self.bin_centers[sampled_indices]  # (batch_size, n_samples)

        # Reshape back to (..., n_samples)
        samples = samples.view(*orig_shape, n_samples)

        return samples

    def _hl_gaussian_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        y_target: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Calculate the cross-entropy loss between the predicted distribution and the target tensor.

        Args:
            logits: Tensor of shape (..., nbins) containing the predicted distribution
            y_target: Tensor of shape (...) containing target distribution
            sigma: Standard deviation for the Gaussian smoothing

        Note: This function constructs a soft target distribution (one per element in y_target)
              by integrating a Gaussian centered at y_i over each bin. (Histogram Loss Gaussian or HL-Gauss)
        """
        assert sigma > 0, "Sigma must be positive."
        assert (
            y_target.shape == logits.shape[:-1]
        ), "y_target must have the same shape as logits except for the last dimension."
        # We'll compute a distribution by integrating the Gaussian cdf in each bin
        # so for bin k, the probability is cdf(upper_edge) - cdf(lower_edge).
        #   cdf(x) = 0.5 * [1 + erf((x - mu) / (sqrt(2)*sigma))]

        # Expand y_target so we can broadcast:
        # y_target: (...) => (..., 1) so we can compare with each bin center
        y_target_expanded = y_target.unsqueeze(-1)  # => (..., 1)

        # We'll use the normal CDF in a piecewise manner:
        def normal_cdf(x, mean, std):
            # cdf = 0.5 * [1 + erf((x - mean)/(sqrt(2)*std))]
            return 0.5 * (1.0 + torch.erf((x - mean) / (math.sqrt(2) * std)))

        # lower and upper edges for each bin, shape (nbins,)
        lower_edges = self.bin_centers - 0.5 * self.bin_width
        upper_edges = self.bin_centers + 0.5 * self.bin_width

        # Now we want to do cdf(upper_edges) - cdf(lower_edges) for each data point
        # We'll broadcast them so that shape => (..., nbins)
        cdf_upper = normal_cdf(upper_edges, y_target_expanded, sigma)
        cdf_lower = normal_cdf(lower_edges, y_target_expanded, sigma)

        # Probability in each bin is difference of cdfs
        p = cdf_upper - cdf_lower  # shape: (..., nbins)

        # Because of numerical issues, it might not sum exactly to 1, so we can renormalize:
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)  # shape: (..., nbins)

        # Now we can compute the cross-entropy loss with the *soft* target:  CE = - sum_k [ p_k * log(q_k) ]
        log_probs = F.log_softmax(logits, dim=-1)  # shape: (..., nbins)
        ce_loss = -torch.sum(p * log_probs, dim=-1).mean(dim=-1)
        return ce_loss

    def _hl_regularized_CRPS_loss(
        self, 
        logits: torch.Tensor,
        y_target: torch.Tensor,
        sigma: float,
        alpha: float = 1,
    ) -> torch.Tensor:
        """Calculates the CRPS loss (https://en.wikipedia.org/wiki/Scoring_rule#Continuous_ranked_probability_score), 
        with an optional parameter, alpha, regulating the strength of the pro-dispersion term.

        Args:
            logits: Tensor of shape (..., nbins) containing the predicted distribution
            y_target: Tensor of shape (...) containing target distribution
            sigma: Standard deviation for the Gaussian smoothing
            alpha: Regulates strength of dispersion term (mean absolute distance) 
                   (0 -> Wasserstein-1 distance, 1 -> standard CRPS)
        """
        assert sigma > 0, "Sigma must be positive."
        assert (
            y_target.shape == logits.shape[:-1]
        ), "y_target must have the same shape as logits except for the last dimension."
        
        y_target_expanded = y_target.unsqueeze(-1)  # => (..., 1)
        def normal_cdf(x, mean, std):
            return 0.5 * (1.0 + torch.erf((x - mean) / (math.sqrt(2) * std)))

        # lower and upper edges for each bin, shape (nbins,)
        lower_edges = self.bin_centers - 0.5 * self.bin_width
        upper_edges = self.bin_centers + 0.5 * self.bin_width

        # Now we want to do cdf(upper_edges) - cdf(lower_edges) for each data point
        # We'll broadcast them so that shape => (..., nbins)
        cdf_upper = normal_cdf(upper_edges, y_target_expanded, sigma)
        cdf_lower = normal_cdf(lower_edges, y_target_expanded, sigma)

        # Probability in each bin is difference of cdfs
        p = cdf_upper - cdf_lower  # shape: (..., nbins)

        # Because of numerical issues, it might not sum exactly to 1, so we can renormalize:
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)  # shape: (..., nbins)

        # Calculate the CRPS loss as sum of Wasserstein + dispersion
        cdf_p = torch.cumsum(p, dim=-1)
        cdf_model = torch.cumsum(torch.softmax(logits, dim=-1), dim=-1)

        wasserstein_part = torch.sum(torch.abs(cdf_model - cdf_p), dim=-1).mean(dim=-1)
        mean_absolute_difference = 2 * torch.sum(cdf_model * (1 - cdf_model), dim=-1).mean(dim=-1)

        return self.bin_width * (wasserstein_part - (alpha / 2.0) * mean_absolute_difference)
    
    def _get_y_shift_scale(
        self, 
        y_context: torch.Tensor
    ):
        """Compute the mean and std of y_context in the observation data, *without* splitting
        into treatment and control groups, since treatment is continuous. 
        """
        y_mean = y_context.mean(dim=1, keepdim=True)
        y_std = y_context.std(dim=1, keepdim=True).clamp(min=1e-8)

        return y_mean, y_std
    
    def _get_t_shift_scale(
        self, 
        t_context: torch.Tensor
    ):
        """Compute the mean and std of t_context in the observation data."""
        t_mean = t_context.mean(dim=1, keepdim=True)
        t_std = t_context.std(dim=1, keepdim=True).clamp(min=1e-8)

        return t_mean, t_std

    def cepo_losses(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor, 
        random_treatments_query: torch.Tensor,
        mu_t_query: torch.Tensor,
        sigma: float | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Calculates loss of predictions on X_query with Uniform[0, 1] treatments.

        The prior already generated random treatments t_query ~ Uniform[0, 1] of shape X.shape[0] 
        and also predicts y_query using the Bernstein polnomial model. We pass this in as mu_t_query, 
        which becomes E_y_target. This is passed to _hl_gaussian_cross_entropy_loss, along with y_pred.
        
        Args:
            X_context (torch.Tensor)
            t_context (torch.Tensor)
            y_context (torch.Tensor)
            X_query (torch.Tensor)
            random_treatments_query (torch.Tensor)
            mu_t_query (torch.Tensor)
            sigma (float | None)
            temperature (float)
        
        Returns:
            torch.Tensor: The loss of the model
        """
        sigma = sigma or self.sigma
        y_shift, y_scale = self._get_y_shift_scale(y_context)
        t_shift, t_scale = self._get_t_shift_scale(t_context)

        # z-standardize the outcomes and treatment
        y_standardized = (y_context - y_shift) / y_scale
        E_y_target = (mu_t_query - y_shift) / y_scale
        t_context_standardized = (t_context - t_shift) / t_scale
        random_treatments_standardized = (random_treatments_query - t_shift) / t_scale

        x_and_t_context = torch.cat(
            [
                t_context_standardized.unsqueeze(-1),
                X_context,
            ],
            dim=2,
        )  # shape: (batch_size,  context_len , num_features + 1)
        x_and_t_query = torch.cat(
            [
                random_treatments_standardized.unsqueeze(-1),
                X_query,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)
        x_and_t = torch.cat(
            [x_and_t_context, x_and_t_query], dim=1
        )  # shape: (batch_size, context_len + query_len, num_features + 1)
        x_src, y_src = self.prepare_input(x_and_t, y_standardized)
        logits = self.model(
            x_src.transpose(0, 1), y_src.transpose(0, 1)
        ).transpose(0, 1)  # shape: (batch_size, query_len, n_bins)
        logits = logits[:, :, -self.model.nbins:]  # only keep the last nbins, which are the predictions
        logits /= temperature  # apply temperature scaling
        return self._hl_gaussian_cross_entropy_loss(
            logits=logits,
            y_target=E_y_target,
            sigma=sigma
        )
        # return self._hl_regularized_CRPS_loss(
        #     logits=logits,
        #     y_target=E_y_target,
        #     sigma=sigma
        # )

    def forward(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        random_treatments_query: torch.Tensor,
        mu_t_query: torch.Tensor,
        sigma: float | None = None,
        temperature: float = 1.0,
    ):
        """The forward method will simply call the cepo_losses method and returns the loss for training.
        This is done to support multi-GPU training.
        """
        return self.cepo_losses(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=X_query,
            random_treatments_query=random_treatments_query,
            mu_t_query=mu_t_query,
            sigma=sigma,
            temperature=temperature,
        )

    def predict_cepo(
        self, 
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        t_query: torch.Tensor,
        temperature: torch.Tensor,  # shape: (num_temperatures, )
        n_samples: int | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Predicts CEPO of (X_query, t_query) given context.

        Args:
            X_context (torch.Tensor): context covariates
            t_context (torch.Tensor): context treatments
            y_context (torch.Tensor): context observed outcomes
            X_query (torch.Tensor): query covariates
            t_query (torch.Tensor): query treatments
            temperature (torch.Tensor)
            n_samples (int | None): samples for bootstrap estimation

        Returns: 
            torch.Tensor: predicted CEPO for (X_query, t_query)
        """
        y_shift, y_scale = self._get_y_shift_scale(y_context)
        t_shift, t_scale = self._get_t_shift_scale(t_context)

        y_standardized = (y_context - y_shift) / y_scale
        t_context_standardized = (t_context - t_shift) / t_scale
        t_query_standardized = (t_query - t_shift) / t_scale

        x_and_t_context = torch.cat(
            [
                t_context_standardized.unsqueeze(-1),
                X_context,
            ],
            dim=2,
        )  # shape: (batch_size,  context_len , num_features + 1)
        x_and_t_query = torch.cat(
            [
                t_query_standardized.unsqueeze(-1),
                X_query,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)

        x_and_t = torch.cat(
            [x_and_t_context, x_and_t_query], dim=1
        )  # shape: (batch_size, context_len + query_len, num_features + 1)

        x_src, y_src = self.prepare_input(x_and_t, y_standardized)

        logits = self.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(
            0, 1
        )  # shape: (batch_size, query_len, nbins)
        logits = logits[:, :, -self.model.nbins :]  # only keep the last nbins, which are the predictions

        logits = logits.unsqueeze(1)  # shape: (batch_size, 1, query_len, nbins)
        y_scale, y_shift = y_scale.unsqueeze(1), y_shift.unsqueeze(1)
        t_query = t_query.unsqueeze(1)  # shape: (batch_size, 1, query_len)

        temperature = temperature[None, :, None, None]  # shape: (1, num_temperatures, 1, 1)

        logits = logits / temperature  # Apply temperature scaling

        mean = self._predict_mean(logits)  # shape: (batch_size, num_temperatures, query_len)
        mean_shift_scaled = mean * y_scale + y_shift

        if n_samples is None:
            return mean_shift_scaled
        
        # shape: (batch_size, num_temperatures, query_len, n_samples)
        samples = self._sample_from_logits(logits, n_samples)
        samples_shift_scaled = samples * y_scale + y_shift
        return mean_shift_scaled, samples_shift_scaled
    
    def get_param_groups(self):
        """
        Return optimizer-specific parameter groups based on the type of the model used.
        This will sometimes help with stabilizing the training.
        """
        if isinstance(self.model, TabDPTLongContextModel):
            return [
                dict(
                    params=self.model.transformer_encoder.parameters(),
                ),
                dict(
                    params=[p for n, p in self.model.named_parameters() if not n.startswith("transformer_encoder")],
                    weight_decay=0.0,  # no weight decay for these params
                ),
            ]
        else:
            return self.model.parameters()
