import torch
import cv2
import os.path as osp
from ultralytics import YOLO

class Predictor(object):
    def __init__(
        self,
        model_path=None,
        model=None,
        device=torch.device("cpu"),
        fp16=False,
    ):
        """
        Args:
            model_path: yolov8权重路径（字符串）
            model: 传入已加载的ultralytics YOLO模型对象（可选）
            device: torch设备，cpu或cuda
            fp16: 是否用fp16推理（目前ultralytics自动管理）
        """
        self.device = device
        self.fp16 = fp16

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = YOLO(model_path)
        else:
            raise ValueError("Must provide model_path or model")

        self.confidence_threshold = 0.0001
        self.nms_threshold = 0.6

    def _resolve_device(self):
        if isinstance(self.device, torch.device):
            return str(self.device)
        if isinstance(self.device, str):
            if self.device in ("gpu", "cuda"):
                return "cuda:0"
            return self.device
        return "cpu"

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        with torch.no_grad():
            timer.tic()
            results = self.model.predict(
                img,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                device=self._resolve_device(),
                half=False,
                verbose=False,
            )

            timer.toc()

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            outputs = None
        else:
            outputs = result.boxes.data.cpu().numpy()

        return [outputs], img_info
