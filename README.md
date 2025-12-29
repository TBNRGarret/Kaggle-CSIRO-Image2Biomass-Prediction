# 🌱 CSIRO Pasture Biomass Prediction

**Computer Vision · Deep Learning · Physics-Informed Learning**

> Fine-tuning a Foundation Vision Transformer (DINOv2 Giant) for large-scale pasture biomass estimation using satellite imagery, with physics-informed constraints and robust data pipelines.


## 1. Project Overview

This project focuses on predicting **pasture biomass components** from high-resolution satellite images, inspired by real-world agricultural AI problems from **CSIRO**.

Unlike standard image regression, this task introduces:

* **Multi-target prediction**
* **Physical consistency constraints**
* **Large-scale Vision Transformer fine-tuning**
* **Data leakage–safe validation strategy**

### 🎯 Prediction Targets

The model predicts **5 biomass-related quantities (grams):**

1. Dry Green Biomass
2. Dry Dead Biomass
3. Dry Clover Biomass
4. Green Dry Matter (GDM)
5. Total Dry Biomass

A physics-informed constraint enforces:

```
Green + Dead + Clover + GDM ≈ Total
```


## 2. Key Technical Highlights

### 🔍 Model

* **Backbone:** `DINOv2 ViT-Giant` (`vit_giant_patch14_reg4_dinov2`)
* **Framework:** PyTorch + timm
* **Input Resolution:** `756 × 1512` (very high resolution)
* **Training Strategy:**

  * Freeze backbone → warm-up head
  * Gradual unfreezing with ultra-low LR
  * Gradient checkpointing to avoid OOM

### 🧠 Why DINOv2 Giant?

* Foundation model trained on massive visual corpora
* Strong generalization on unseen satellite imagery
* Token-based global pooling preserves spatial context


## 3. Dataset & Data Pipeline

### 📁 Dataset Structure

* Images + metadata stored in a **long-format CSV**
* Converted into **wide-format supervised targets** via pivoting

### 🛡️ Preventing Data Leakage

A **Stratified Group K-Fold** strategy is used:

* **Group:** `Sampling_Date`
* **Stratify:** `State`
* Ensures:

  * No temporal leakage
  * Geographic generalization
  * Realistic validation performance

```text
Train / Validation split respects:
- Location (State)
- Time (Sampling Date)
```


## 4. Data Augmentation

Implemented with **Albumentations**:

* Resize to target resolution
* Horizontal & vertical flips
* Color jitter (brightness & contrast)
* ImageNet normalization

```python
A.Compose([
    Resize,
    Flip,
    ColorJitter,
    Normalize,
    ToTensorV2
])
```

This improves robustness against:

* Illumination changes
* Seasonal variation
* Sensor noise


## 5. Physics-Informed Loss Function

### ⚖️ CSIRO_ConsistencyLoss

Instead of naive regression loss, the model uses a **custom loss** combining:

1. **Component-wise MAE**
2. **Physical consistency penalty**

```math
Loss = MAE(pred, target) + 0.5 × MAE(sum(components), total)
```

This enforces **domain knowledge** directly into training, improving:

* Stability
* Interpretability
* Real-world reliability


## 6. Training Strategy

### ⚙️ Optimization

* **Optimizer:** AdamW
* **Schedulers:** CosineAnnealingWarmRestarts
* **Mixed Precision:** AMP (FP16)
* **Gradient Accumulation:** Enabled
* **Gradient Clipping:** max_norm = 1.0

### 🧊 Two-Phase Training

| Phase   | Description                            |
| ------- | -------------------------------------- |
| Phase 1 | Freeze backbone, train regression head |
| Phase 2 | Unfreeze backbone with very small LR   |

This prevents catastrophic forgetting on a giant foundation model.


## 7. Evaluation Metric

Primary metric: **Mean Absolute Error (MAE)**

* Reported per target
* Final score = average MAE across all components

Example output:

```
Green: 12.3g | Dead: 15.1g | Total: 18.7g
```

MAE is preferred over MSE for robustness against biomass outliers.
