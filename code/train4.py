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
from ultralytics import YOLO
from ultralytics.data.augment import Compose
from ultralytics.data import build_dataloader, build_yolo_dataset

# -------------------------- Enhanced Preprocessor --------------------------
import cv2
import numpy as np

class AdvancedPreprocessor:
    def __init__(self, clahe_limit=4.0, gamma_range=(0.3, 1.7), noise_std=5):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
        self.gamma_range = gamma_range
        self.noise_std = noise_std

    def __call__(self, img):
        # Ensure img is uint8 for OpenCV compatibility
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # LAB conversion
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L
        l = self.clahe.apply(l)

        # Gamma correction
        gamma = np.random.uniform(*self.gamma_range)
        l = np.power(l / 255.0, gamma) * 255.0
        l = l.astype(np.uint8)

        # Noise (30% chance)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, l.shape).astype(np.int16)
            l = np.clip(l.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Merge and back to RGB
        processed_lab = cv2.merge([l, a, b])
        rgb = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB)

        # Return as float32 in [0,1] for YOLO
        return rgb.astype(np.float32) / 255.0


# -------------------------- Optimized Hyperparameters --------------------------
EPOCHS = 30
MOSAIC = 1.0  # Use full mosaic augmentation
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False
PREPROCESS_PROB = 0.85  # Higher probability for preprocessing

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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='yolov8m-seg.pt')  # Upgraded model
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Initialize advanced preprocessor
    preprocessor = AdvancedPreprocessor(clahe_limit=4.0, gamma_range=(0.3, 1.7), noise_std=7)

    # -------------------------- Enhanced Dataset Builder --------------------------
    def custom_build_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
        dataset = build_yolo_dataset(cfg, img_path, batch, data, mode, rect, stride)
        
        original_transforms = dataset.transforms
        def new_transforms(im, labels, *args, **kwargs):
            if mode == 'train' and np.random.random() < PREPROCESS_PROB:
                im = preprocessor(im)
            return original_transforms(im, labels, *args, **kwargs)
            
        dataset.transforms = new_transforms
        return dataset

    # -------------------------- Load Upgraded Model --------------------------
    model = YOLO(os.path.join(this_dir, 'yolov8m.pt'))  # Use larger model architecture
    
    # Override dataset builder
    model.train_loader = lambda *args, **kwargs: build_dataloader(
        *args, **kwargs, dataset_func=custom_build_dataset
    )

    # -------------------------- Train with Optimized Settings --------------------------
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
        cls=0.5,
        dfl=1.5,
        mosaic=args.mosaic,
        mixup=0.3,
        copy_paste=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=0.1,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.3,
        close_mosaic=15,
        patience=15,
        label_smoothing=0.1,
        nbs=64,
        overlap_mask=True
    )
