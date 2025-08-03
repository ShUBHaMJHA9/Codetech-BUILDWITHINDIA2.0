
# 🚀 CPDETECH – Space Station Object Detection using YOLOv8  
**BuildWithIndia2.0 Hackathon – Duality AI | Falcon Digital Twin**

![Results](model_weights/train2/results.png)

---

## 👨‍💻 Team Members

| Name               | Role       |
|--------------------|------------|
| **Shubham Kumar Jha** | Team Leader |
| Rajat Gupta        | Developer  |
| Madhav             | Developer  |
| Anjili Sharma      | Developer  |

---

## 📌 Project Overview

**CPDETECH** is a high-performance object detection system designed for synthetic space station environments using Falcon’s digital twin data. The model detects:

- 🔧 Toolbox  
- 🧯 Fire Extinguisher  
- 🪫 Oxygen Tank  

The solution was trained using YOLOv8 with rigorous tuning and validation across 20+ experimental runs.

---

## 📁 Project Structure

```
├── CODETECH.ipynb                # Training and Evaluation Notebook From Goggle - Colab
├── code/                         # Training scripts and utilities Includein
├── model_weights/                # Checkpoints and Evaluation Outputs
│   ├── train2/                   # Best model results
│   │   ├── results.png
│   │   ├── confusion_matrix.png
│   │   ├── weights/              # Trained .pt files
│   └── val/                      # Validation predictions
    
```

---
## 📊 Training Results Summary

We trained over **40 models** to optimize performance. Below are the top 5 best-performing models:

| Model Name | Train Accuracy (%) | Prediction Accuracy (mAP@0.5) (%) | Notes                          |
|------------|--------------------|----------------------------------|--------------------------------|
| `train`   | 95.3               | 89.2                             | Baseline training              |
| `train2`   | 95.1               | 88.9                             | Improved with better mosaic    |
| `train3`  | 88.2               | 85.4                             | Tuned learning rate            |
| `train4`  | **95.8**           | **89.7**                         | 🏆 Best overall performance     |
| `train5` | 95.1               | 88.5                             | Focused on critical targets    |
## 🧠 Model Architecture

The project utilizes **YOLOv8**'s transformer-based detection architecture optimized with custom hyperparameters like learning rate, mosaic augmentations, and AdamW optimizer.

## 📌 Key Features

- ✅ Custom dataset support
- 📈 High training accuracy (~95.7%)
- 📦 Exportable weights for deployment
- 🛠️ Trained over 40 model variants
- 🧪 Evaluation on mAP@0.5 and object recall

## 📷 Example Detections

*(Include screenshots or result visualizations if available here)*

## 📁 Model Weights

You can find all trained weights inside the `model_weights/` folder.

## 📒 Training Notebook

Check `CODETECH.ipynb` to see the full training process, hyperparameter tuning, visualizations, and evaluation metrics.

---
## 📈 Visual Results

| Preview                      | File Path                                    |
|-----------------------------|-----------------------------------------------|
| 📉 Loss / mAP Curve         | `model_weights/train2/results.png`            |
| 🔲 Confusion Matrix         | `model_weights/train2/confusion_matrix.png`   |
| 📸 Training Batches         | `model_weights/train2/train_batch0.jpg`       |
| 🧪 Validation Predictions    | `model_weights/train2/val_batch0_pred.jpg`    |
The project utilizes **YOLOv8**'s transformer-based detection architecture optimized with custom hyperparameters like learning rate, mosaic augmentations, and AdamW optimizer.

## ⚙️ Usage Instructions

### 🔧 Requirements

```bash
pip install ultralytics opencv-python matplotlib
```

### 🚀 Training

```bash
python train.py
```

### 🔍 Prediction

```bash
python predict.py --weights model_weights/train2/weights/best.pt
```

---

## 📄 Report

📥 [SMARTCROWD_Hackathon_Report.pdf](./SMARTCROWD_Hackathon_Report.pdf)

---

## 🔗 Maintainer

> 👤 [@ShUBHaMJHA9](https://github.com/ShUBHaMJHA9)

---

🛰️ *"We detect the future, in space and beyond."*
