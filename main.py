## hydra imports
import hydra
from omegaconf import DictConfig

## modules
import srcs.Trainer, srcs.Setting

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")

def mission(cfg: DictConfig):
    srcs.Trainer.run(cfg)

if __name__ == "__main__":
    mission()


