import sys
from pathlib import Path
import torch
from hydra import compose, initialize
from PIL import Image

from yolo.tools.solver import InferenceModel
from yolo import AugmentationComposer, draw_bboxes

class DetectionInfer(InferenceModel):
    def __init__(
        self,
        config_path="yolo/config",
        config_name="config",
        model_name="v9-s",
        class_num=4,
        custom_model_path="ema_cleaned.pt",
        device="cuda:0",
        class_list=None,
    ):
        with initialize(config_path=config_path, version_base=None, job_name="infer_job"):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    "task=inference",
                    f"model={model_name}",
                    f"dataset.class_num={class_num}",
                ]
            )
            cfg.task.nms.min_confidence = 0.1
            cfg.task.nms.min_iou = 0.1
            if class_list:
                cfg.dataset.class_list = class_list
            self.cfg = cfg

        self.cfg.weight = custom_model_path
        self._device = torch.device(device)
        super().__init__(self.cfg)
        self.model.eval()
        self.model.to(self._device)
        self.transform = AugmentationComposer([], self.cfg.image_size)
        self.setup(stage=None)

    def predict(self, image_path, output_path=None):
        pil_image = Image.open(image_path)
        image, _, rev_tensor = self.transform(pil_image)
        image = image.to(self._device)[None]
        rev_tensor = rev_tensor.to(self._device)[None]
        with torch.no_grad():
            predicts = self.post_process(self.model(image), rev_tensor=rev_tensor)
        
        print(f"Predictions: {predicts}")
        output_image = draw_bboxes(
            pil_image,
            predicts,
            idx2label=getattr(self.cfg.dataset, "class_list", None)
        )
        if output_path:
            output_image.save(output_path)
            print(f"Output image saved at: {output_path}")
        return output_image


if __name__ == "__main__":
    # Example usage
    detector = DetectionInfer(
        config_path="yolo/config",
        config_name="config",
        model_name="v9-t",
        class_num=4,
        custom_model_path="best.ckpt",
        device="cuda:0",
        class_list=['buffalo', 'elephant', 'rhino', 'zebra'],
    )
    detector.predict("test.jpg", "output_new.jpg")
