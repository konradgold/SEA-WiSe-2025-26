from hydra import initialize, compose
from omegaconf import DictConfig
from sea.utils.config import MainConfig


def Config(load: bool) -> DictConfig:
    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(config_name="base")
    return cfg if load else MainConfig()