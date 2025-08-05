from hydra import compose, initialize
from lightning import Trainer
from yolo.tools.solver import TrainModel
from typing import Optional
import yaml

class DetectionTrainer:
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        config_path: Optional[str] = "yolo/config",
        config_name: Optional[str] = "config",
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
        self.trainer = Trainer(
            accelerator="auto", 
            max_epochs=self.cfg.task.epoch,
            precision="16-mixed", 
            logger=True,
            log_every_n_steps=1,
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            deterministic=True,
            enable_progress_bar=True
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
