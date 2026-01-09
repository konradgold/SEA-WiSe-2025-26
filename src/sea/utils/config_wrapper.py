from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from sea.utils.config import MainConfig


def Config(load: bool = True) -> DictConfig:
    if not load:
        return MainConfig()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(config_name="base")
    return cfg
