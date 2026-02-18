import hydra
from omegaconf import DictConfig

from supcon.utils import patch_hydra_argparse
from train_pretrain import run_pretrain

patch_hydra_argparse()


@hydra.main(config_path="configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    run_pretrain(cfg)


if __name__ == "__main__":
    main()
