from omegaconf import OmegaConf
from pathlib import Path
from qiskit import IBMQ
from src.runners import S_Runner, LOSO_Runner
import wandb
import warnings

#config 파일을 불러오는 경로 설정
CONFIG_DIR = Path("Configs")

#Omegaconf를 이용해서 저장된 config 파일을 불러옴
def main(cfg=OmegaConf.load(CONFIG_DIR / "config.yaml")) -> None:
    #Omegaconf를 이용해서 저장된 model에 관한 config 파일을 불러옴
    model_params = OmegaConf.load(CONFIG_DIR / "models.yaml")
    #두 config 파일을 합침
    cfg = OmegaConf.merge(cfg, model_params)
    #명령어로 입력한 내용과 config를 합침
    cfg.merge_with_cli()

    #wandb에 저장될 위치와 이름을 정함
    wandb.init(entity="qubit-hanyang", project=f"{cfg.log.project_name}", name=f"{cfg.Simple_QHN.model} {cfg.optimizer.lr},{cfg.Simple_QHN.shift} {cfg.Simple_QHN.backend}")
    #IBMQ 계정에 로그인, 처음 사용할 경우 save_account 필요
    #IBMQ.save_account("token")
    IBMQ.load_account()
    #경고 메시지 무시
    warnings.filterwarnings(action='ignore')

    #원하는 종류의 runner를 고르고 config파일에서 변수들을 불러와서 실행
    runner = S_Runner(
        log=cfg.log,
        optimizer=cfg.optimizer,
        loader=cfg.loader,
        network=cfg.network,
        data=cfg.data,
    )
    metrics = runner.run(profiler=cfg.get("profiler", "simple"))


if __name__ == "__main__":
    
    main()
