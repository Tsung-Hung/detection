import torch
from hydra import compose, initialize
from yolo.tools.solver import TrainModel
from lightning import Trainer

class DetectionTrainer:
    def __init__(
        self,
        config_path="yolo/config",
        config_name="config",
        model_name="v9-s",
        dataset_name=None,
        class_num=4,
        weight_path=None,
        device="cuda:0",
        batch_size=None,
        epochs=100,
    ):
        with initialize(config_path=config_path, version_base=None, job_name="train_job"):
            overrides = [
                "task=train",
                f"model={model_name}",
                f"dataset.class_num={class_num}",
            ]
            if dataset_name:
                overrides.append(f"dataset={dataset_name}")
            if weight_path:
                overrides.append(f"weight={weight_path}")

            cfg = compose(
                config_name=config_name,
                overrides=overrides
            )
            if batch_size:
                cfg.task.data.batch_size = batch_size
            
            cfg.task.epoch = epochs
            
            self.cfg = cfg

        self.device = torch.device(device)
        self.model = TrainModel(self.cfg)
        self.trainer = Trainer(
            accelerator="auto", 
            precision="16-mixed", 
            logger=True,
            enable_progress_bar=True
        )

    def train(self):
        self.trainer.fit(self.model)

if __name__ == "__main__":
    trainer = DetectionTrainer(
        config_path="yolo/config",
        config_name="config",
        model_name="v9-t",
        dataset_name="africanwildlife",
        class_num=4,
        weight_path=None,
        device="cuda:0",
        batch_size=8,
        epochs=10,
    )
    trainer.train()