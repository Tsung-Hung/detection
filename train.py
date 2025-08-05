from hydra import compose, initialize
from lightning import Trainer
from allinvision_detection.yolo.tools.solver import TrainModel

class DetectionTrainer:
    def __init__(
        self,
        config_path="yolo/config",
        config_name="config",
        model_name="v9-s",
        dataset_name=None,
        class_num=4,
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

            cfg = compose(
                config_name=config_name,
                overrides=overrides
            )
            if batch_size:
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
        dataset_name="africanwildlife",
        class_num=4,
        batch_size=32,
        epochs=10,
    )
    trainer.train()
