from omegaconf import OmegaConf
from pathlib import Path
from qiskit import IBMQ
from src.runners import S_Runner

CONFIG_DIR = Path("Configs")


def main(cfg=OmegaConf.load(CONFIG_DIR / "config.yaml")) -> None:
    model_params = OmegaConf.load(CONFIG_DIR / "models.yaml")
    cfg = OmegaConf.merge(cfg, model_params)
    cfg.merge_with_cli()

    runner = S_Runner(
        log=cfg.log,
        optimizer=cfg.optimizer,
        loader=cfg.loader,
        network=cfg.network,
        data=cfg.data,
    )
    metrics = runner.run(profiler=cfg.get("profiler", "simple"))


if __name__ == "__main__":
    #IBMQ.save_account("39e7cec1ae590142d86caa525b2bf85bc5fbc8893ba6b8a8bd7de88582afdff10c0ec41104c54851efe419e0cf125528184d9c01091878a8617ca41cb2754176")
    main()
