from hydra import compose, initialize
from lightning import Trainer
from yolo.tools.solver import TrainModel
from typing import Optional
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
    ):
        with initialize(config_path=config_path, version_base=None, job_name="train_job"):
            overrides = [
                "task=train",
                f"model={model_name}",
            ]

            cfg = compose(
                config_name=config_name,
                overrides=overrides
            )

            if dataset_path:
                with open(dataset_path, 'r') as f:
                    user_dataset_dict = yaml.safe_load(f)

                if user_dataset_dict:
                    for key, value in user_dataset_dict.items():
                        cfg.dataset[key] = value

            cfg.task.data.batch_size = batch_size
            cfg.task.epoch = epochs
            self.cfg = cfg

        self.model = TrainModel(self.cfg)
        self.trainer = self._get_trainer()

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
        config_name="config",
        model_name="v9-t",
        dataset_path="C:/Users/tsung-hung/Downloads/YOLO/africanwildlife.yaml",
        batch_size=16,
        epochs=10,
    )
    print(trainer.cfg)
    # trainer.train()
