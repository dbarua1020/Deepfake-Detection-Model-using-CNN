# 🧠 Deepfake-Detection-Model-using-CNN-Convolutional-Neural-Networks

This repository contains a deep learning-based solution for detecting deepfake videos using Convolutional Neural Networks (CNNs), with a focus on the EfficientNetB0 architecture. The system leverages face detection, data augmentation, and transfer learning to accurately classify video frames as real or fake.

## 📁 Dataset

- **Source**: [FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics)
- **Structure**: Real and fake video folders with `.mp4` files.
- **Preprocessing**:
  - Frame extraction every 30 frames.
  - Face detection using Haar Cascades.
  - Cropping, resizing (224x224), and normalization.
  - Data augmentation applied to fake frames (e.g., rotation).

## 🧪 Model Architecture

- **Base**: [EfficientNetB0](https://arxiv.org/abs/1905.11946) (pre-trained on ImageNet)
- **Custom Head**:
  - Global Average Pooling
  - Batch Normalization
  - Dense Layer (256 units, ReLU, L2 regularization)
  - Dropout
  - Output: Sigmoid activation for binary classification

## ⚙️ Training Configuration

- **Optimizer**: Adam (`learning_rate=0.0001`)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Callbacks**:
  - EarlyStopping
  - ReduceLROnPlateau

## 📊 Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | ~96.8%    |
| Precision  | ~99%      |
| F1 Score   | ~96%      |
| AUC        | ~96%      |

- Inference speed: ~15–20 FPS on a mid-tier GPU.
- Suitable for near real-time deployment.

## 📈 Visualizations

- Training vs Validation Accuracy
- Training vs Validation Loss

![Model Accuracy](./images/accuracy_plot.png)
![Model Loss](./images/loss_plot.png)

## 🧠 Features

- Face-based deepfake detection
- Transfer learning with EfficientNetB0
- Data augmentation for improved generalization
- Modular, scalable architecture
- Real-time processing capability

## 💾 Model Export

The final trained model is saved in:
```bash
improved_deepfake_model.keras
