# Vision Transformer (ViT) for CIFAR-10 Classification

This repository implements a **Vision Transformer (ViT)** from scratch using **PyTorch** to classify images from the CIFAR-10 dataset. The model integrates modern training techniques such as enhanced data augmentation, label smoothing, cosine annealing learning rate scheduling with warm-up, gradient clipping, and visualization of attention maps and predictions.

> **Achieved Test Accuracy:** `84%` on the CIFAR-10 test set.

---

##  Dataset

- **CIFAR-10**: A standard benchmark dataset for image classification consisting of 60,000 32x32 color images in 10 classes (6,000 images per class).
- Split: 50,000 training images and 10,000 test images.

---

## Model Architecture

The model is a customized **Vision Transformer** built with the following key components:

- **Patch Embedding**: Converts image into non-overlapping patches.
- **Multi-head Self-Attention**: Captures spatial relationships across patches.
- **Transformer Encoder**: Stack of attention + feedforward layers (12 layers, 8 heads).
- **Classification Head**: Linear layer on the `[CLS]` token for classification.

---

##  Features

- ViT implementation in PyTorch  
- Cosine learning rate scheduling with warm-up  
- Label smoothing in cross-entropy loss  
- Gradient clipping for stability  
- Enhanced data augmentation pipeline  
- Attention map visualization per layer  
- Prediction confidence display  
- Custom colormaps for insightful plots

---

## Results

| Metric         | Value   |
|----------------|---------|
| Test Accuracy  | **84%** |
| Epochs         | 100     |
| Optimizer      | AdamW   |
| Embedding Dim  | 384     |
| Transformer Depth | 12  |

---

