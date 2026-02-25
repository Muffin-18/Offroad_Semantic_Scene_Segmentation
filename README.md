# Offroad Semantic Scene Segmentation
## Duality AI – Hack for Green Bharat 2026
### Team Decipher

---

## Overview

This project presents a Vision Transformer-based semantic segmentation framework designed to classify off-road terrain environments using synthetic digital twin data.

The model performs pixel-wise classification of complex off-road scenes into multiple terrain categories and is optimized to maximize Mean Intersection over Union (mIoU).

The system combines a pretrained Vision Transformer backbone with a convolutional decoder to achieve stable convergence and competitive segmentation performance.

---

## Team Members

- Asmit Kumar Jena (Team Leader)
- Subham Dutta
- Abhishek Dhal
- Ananya Raj

---

## Problem Statement

Autonomous systems operating in unstructured environments require accurate terrain understanding. Unlike urban roads, off-road scenes contain diverse vegetation, obstacles, and natural elements.

This project aims to build a robust semantic segmentation model capable of classifying terrain pixels into meaningful categories such as vegetation, sky, obstacles, and landscape.

---

## Dataset

The dataset consists of synthetic digital twin off-road environments.

### Training Set
- 2857 RGB images
- 2857 segmentation masks

### Validation Set
- 317 RGB images
- 317 segmentation masks

### Classes
- Sky
- Landscape
- Trees
- Lush Bushes
- Dry Grass
- Dry Bushes
- Logs
- Rocks
- Ground Clutter
- Background

---

## Model Architecture

### Backbone
- DINOv2 Vision Transformer (ViT-S/14)
- Patch size: 14
- Self-supervised pretrained
- Backbone frozen during training

### Segmentation Head
- ConvNeXt-inspired decoder
- Upsampling layers
- Batch normalization
- Final 1×1 convolution for class prediction

### Design Rationale
- Transformers capture global contextual information
- CNN decoder restores spatial resolution
- Frozen backbone ensures stable training

---

## Training Configuration

- Optimizer: AdamW
- Learning Rate Scheduler: Cosine Annealing
- Loss Function:
  - Weighted Cross Entropy
  - Dice Loss
- Mixed Precision Training: Enabled
- Early Stopping: Implemented
- Hardware Used: NVIDIA RTX 4060

---

## Results

### Best Validation Metrics

| Metric | Value |
|--------|--------|
| Mean IoU | 0.3670 |
| Dice Score | 0.5322 |
| Pixel Accuracy | 0.7101 |
| Best Epoch | 25 |

The model demonstrated stable convergence with minimal train–validation gap.

---

## Per-Class IoU (Best Epoch)

| Class | IoU |
|--------|--------|
| Sky | 0.9704 |
| Landscape | 0.5039 |
| Background | 0.4542 |
| Dry Grass | 0.4350 |
| Trees | 0.3978 |
| Lush Bushes | 0.3928 |
| Dry Bushes | 0.1874 |
| Rocks | 0.1741 |
| Ground Clutter | 0.1217 |
| Logs | 0.1165 |

### Observations

- Strong performance on large-area classes (Sky, Landscape)
- Small-object segmentation remains challenging
- Texture similarity affects vegetation categories

---
