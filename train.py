# config
EPOCHS = 40
MOSAIC = 0.64
OPTIMIZER = 'AdamW'
MOMENTUM = 0.201
LR0 = 0.0008
LRF = 0.0085
SINGLE_CLS = False
IMGSZ = 640
BATCH = 16

# imports
import argparse
import os
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")

    # arguments
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation probability')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum for optimizer')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Train as single class')
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    parser.add_argument('--batch', type=int, default=BATCH, help='Batch size')

    args = parser.parse_args()

    # working directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # load model
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    # train
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device='cpu',  # explicitly use CPU
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        single_cls=args.single_cls
    )
