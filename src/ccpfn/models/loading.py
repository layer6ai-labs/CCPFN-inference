import torch
from omegaconf import OmegaConf

from .model import TabDPTLongContextModel

_MODEL_CONFIG_DEFAULTS = {
    "max_num_covariates": 100,
    "treatment_dim": 1,
    "nbins": 64,
}

def _cfg_to_dict(cfg) -> dict:
    """Convert OmegaConf config (possibly a struct) to a plain Python dict."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


class DictToObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)


def load_pretrained_tabdpt_model(ckpt_path: str | None, ckpt: dict | None = None, **overrides) -> TabDPTLongContextModel:
    assert ckpt_path is not None or ckpt is not None, "Either ckpt_path or ckpt must be provided."
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu") if ckpt is None else ckpt
    cfg_dict = _cfg_to_dict(checkpoint["cfg"])
    for key, default in _MODEL_CONFIG_DEFAULTS.items():
        cfg_dict["model"].setdefault(key, default)
    for key, value in overrides.items():
        cfg_dict["model"][key] = value

    config = DictToObject(cfg_dict)
    model = TabDPTLongContextModel.load(
        model_state=checkpoint["model"],
        config=config,
    )
    return model


def load_pretrained_tabdpt_config(ckpt_path: str | None, ckpt: dict | None = None, **overrides) -> dict:
    assert ckpt_path is not None or ckpt is not None, "Either ckpt_path or ckpt must be provided."
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu") if ckpt is None else ckpt
    cfg_dict = _cfg_to_dict(checkpoint["cfg"])
    for key, default in _MODEL_CONFIG_DEFAULTS.items():
        cfg_dict["model"].setdefault(key, default)
    for key, value in overrides.items():
        cfg_dict["model"][key] = value

    return cfg_dict
