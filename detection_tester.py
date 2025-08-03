import torch
from hydra import compose, initialize
from yolo.tools.solver import ValidateModel
from lightning import Trainer

class DetectionTester:
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
    ):
        with initialize(config_path=config_path, version_base=None, job_name="test_job"):
            overrides = [
                "task=validation",
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
            self.cfg = cfg

        self.device = torch.device(device)
        self.model = ValidateModel(self.cfg)
        self.trainer = Trainer(accelerator="auto", precision="16-mixed", logger=False, enable_progress_bar=False)

    def test(self):
        results = self.trainer.validate(self.model)
        # mAP 結果通常在 results[0] 字典中
        if results and isinstance(results, list) and isinstance(results[0], dict):
            map_50 = results[0].get("map_50", None)
            map_5095 = results[0].get("map", None)
            print(f"mAP@0.5: {map_50}, mAP@0.5:0.95: {map_5095}")
            return results[0]
        return results

if __name__ == "__main__":
    tester = DetectionTester(
        config_path="yolo/config",
        config_name="config",
        model_name="v9-t",
        dataset_name="africanwildlife",
        class_num=4,
        weight_path="best.ckpt",
        device="cuda:0",
        batch_size=None,
    )
    results = tester.test()