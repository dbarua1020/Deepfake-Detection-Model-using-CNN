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

<img width="296" alt="image" src="https://github.com/user-attachments/assets/19e46b49-61c7-4b38-b7ce-0ea5ad720a4d" />

- **Base**: [EfficientNetB0](https://arxiv.org/abs/1905.11946) (pre-trained on ImageNet)
- **Custom Head**:
  - Global Average Pooling
  - Batch Normalization
  - Dense Layer (256 units, ReLU, L2 regularization)
  - Dropout
  - Output: Sigmoid activation for binary classification

<img width="301" alt="image" src="https://github.com/user-attachments/assets/d965cbf0-3dad-4e0e-be00-f173de600f87" />


<img width="300" alt="image" src="https://github.com/user-attachments/assets/6acbb27c-3dcb-46ff-ab7f-49847d26b5b6" />

Comparison of different models

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

## 📄 Research Publication

As part of our academic contributions, we have authored a research paper titled **"Deepfake Detection using Convolutional Neural Networks (CNN)"**, which was submitted to the **Sixth International Conference on Computing Communication and Industry Standard** on **12th May 2025** (Paper ID: 21). 

The paper, currently under review, presents our methodology for building an efficient deepfake detection system using CNNs, with a focus on preserving digital media integrity. This submission reflects our commitment to advancing research in AI and cybersecurity.

