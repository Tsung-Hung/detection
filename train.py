from hydra import compose, initialize
from lightning import Trainer
from yolo.tools.solver import TrainModel
from typing import Optional
from omegaconf import OmegaConf, DictConfig
import yaml

class DetectionTrainer:
    DEFAULT_CONFIG_PATH = "yolo/config"
    DEFAULT_CONFIG_NAME = "config"
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        model_name: Optional[str] = "v9-s",
        batch_size: Optional[int] = 16,
        epochs: Optional[int] = 100,
        config_path: Optional[str] = None,
    ):
        self.cfg = self._build_config(
            dataset_path=dataset_path,
            model_name=model_name,
            batch_size=batch_size,
            epochs=epochs,
            config_path=config_path
        )

        self.model = TrainModel(self.cfg)
        self.trainer = self._get_trainer()

    def _build_config(
        self,
        dataset_path: str,
        model_name: str,
        batch_size: int,
        epochs: int,
        config_path: Optional[str],
    ) -> DictConfig:
        """
        """
        cfg_path = config_path or self.DEFAULT_CONFIG_PATH

        try:
            with initialize(config_path=cfg_path, version_base=None, job_name="train_job"):
                overrides = [
                    "task=train",
                    "dataset=default",
                    f"model={model_name}",
                ]
                cfg = compose(config_name=self.DEFAULT_CONFIG_NAME, overrides=overrides)

            user_dataset_cfg = OmegaConf.load(dataset_path)
            cfg.dataset = OmegaConf.merge(cfg.dataset, user_dataset_cfg)

            cfg.task.data.batch_size = batch_size
            cfg.task.epoch = epochs
            return cfg

        except FileNotFoundError:
            logging.error(f"錯誤：找不到資料集設定檔於 '{dataset_path}'")
            raise
        except Exception as e:
            logging.error(f"建立設定檔時發生未知錯誤: {e}")
            raise

    def _get_trainer(self):
        trainer_params = self.cfg.task.get("trainer", {})
        return Trainer(
            accelerator=trainer_params.get("accelerator", "auto"),
            max_epochs=self.cfg.task.epoch,
            precision=trainer_params.get("precision", "16-mixed"),
            logger=trainer_params.get("logger", True),
            log_every_n_steps=trainer_params.get("log_every_n_steps", 1),
            gradient_clip_val=trainer_params.get("gradient_clip_val", 10),
            gradient_clip_algorithm=trainer_params.get("gradient_clip_algorithm", "norm"),
            deterministic=trainer_params.get("deterministic", True),
            enable_progress_bar=trainer_params.get("enable_progress_bar", True)
        )

    def train(self):
        self.trainer.fit(self.model)

if __name__ == "__main__":
    # Example usage
    trainer = DetectionTrainer(
        config_path="yolo/config",
        model_name="v9-t",
        dataset_path="C:/Users/tsung-hung/Downloads/YOLO/africanwildlife.yaml",
        batch_size=16,
        epochs=10,
    )
    print(trainer.cfg)
    # trainer.train()
