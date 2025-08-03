"""
-------------------------------------------------------
  PROJECT NAME : CPDETECH – Space Station Object Detection using YOLOv8
  HACKATHON    : BuildWithIndia2.0 by Duality AI
  TEAM NAME    : CODETECH
  DESCRIPTION  : AI-powered object detection system trained using Falcon synthetic space station data.

  TEAM MEMBERS :
    - Shubham Kumar Jha (Team Leader) - @ShUBHaMJHA9
    - Rajat Gupta
    - Madhav
    - Anjili Sharma

  REPO LINK    : https://github.com/ShUBHaMJHA9/Codetech-BUILDWITHINDIA2.0
  DATE         : August 2025

  NOTE         : This project uses YOLOv8 for object detection and has been evaluated using mAP, precision, recall,
                 and visual validation from the Falcon dataset. All scripts, models, and visualizations are organized
                 for easy reproducibility and deployment.

-------------------------------------------------------
"""


import argparse
import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset

# -------------------------- Optimized Hyperparameters --------------------------
EPOCHS = 40
MOSAIC = 0.75
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False
PREPROCESS_PROB = 0.65

# -------------------------- Focused Preprocessing --------------------------
class FocusedPreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.gamma_range = (0.4, 1.6)
        self.noise_intensity = 0.002

    def apply_gamma(self, img, gamma):
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def add_noise(self, img):
        noise = np.random.normal(0, self.noise_intensity * 255, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    def __call__(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if random.random() > 0.4:
            gamma = random.uniform(*self.gamma_range)
            img = self.apply_gamma(img, gamma)

        if random.random() > 0.7:
            img = self.add_noise(img)

        return img

# -------------------------- Training Logic --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mosaic', type=float, default=MOSAIC)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--lr0', type=float, default=LR0)
    parser.add_argument('--lrf', type=float, default=LRF)
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')  # Use first GPU (GPU:0)
 # default to CPU for Colab compatibility
    parser.add_argument('--model', type=str, default='yolov8m.pt')
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    os.chdir(this_dir)

    preprocessor = FocusedPreprocessor()

    def custom_build_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
        dataset = build_yolo_dataset(cfg, img_path, batch, data, mode, rect, stride)
        original_transforms = dataset.transforms

        def new_transforms(im, labels, *args, **kwargs):
            if mode == 'train' and random.random() < PREPROCESS_PROB:
                im = preprocessor(im)
            return original_transforms(im, labels, *args, **kwargs)

        dataset.transforms = new_transforms
        return dataset

    model = YOLO(os.path.join(this_dir, args.model))
    model.train_loader = lambda *a, **kw: build_dataloader(*a, **kw, dataset_func=custom_build_dataset)

    results = model.train(
    data=os.path.join(this_dir, "yolo_params.yaml"),
    epochs=args.epochs,
    imgsz=args.imgsz,
    device=args.device,
    single_cls=args.single_cls,
    optimizer=args.optimizer,
    lr0=args.lr0,
    lrf=args.lrf,
    momentum=args.momentum,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.8,
    dfl=1.0,
    mask_ratio=4,
    mosaic=args.mosaic,
    mixup=0.1,
    copy_paste=0.05,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=8.0,
    translate=0.08,
    scale=0.5,
    shear=2.0,
    perspective=0.00025,
    fliplr=0.4,
    flipud=0.1,
    auto_augment='rand-m7-mstd0.5',
    close_mosaic=20,
    # label_smoothing=0.05,  # <-- optional: remove to avoid warning
    nbs=64,
    overlap_mask=True,
    deterministic=True,
    plots=True,
    patience=20
)

