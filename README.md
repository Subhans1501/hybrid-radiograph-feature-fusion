# Hybrid Deep & Handcrafted Feature Fusion for Chest Radiograph Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical computer vision framework benchmark investigating whether fusing classic multi-scale handcrafted descriptors (**HOG**, **Multi-Scale Uniform LBP**, **24-Kernel Gabor Filter Banks**, and **SIFT-BoVW with TF-IDF**) into modern deep neural backbones (**ConvNeXt-Tiny** and **Swin Transformer-T**) enhances classification performance, feature calibration, and interpretability on pediatric chest radiographs.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |         Chest Radiograph Input        |
                      |            (256 x 256 Gray)           |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Symmetric CLAHE Enhancement     |
                      |    (clipLimit=2.0, tileGrid=(8, 8))   |
                      +---------+-------------------+---------+
                                |                   |
        +-----------------------+                   +-----------------------+
        |                                                                   |
        v (224 x 224 RGB)                                                   v (128 x 128 Gray)
+-------------------------------+                                   +-------------------------------+
|     Deep Backbone Stream      |                                   |  Handcrafted Feature Stream   |
|   - ConvNeXt-Tiny (Stage 3)   |                                   |   - Multi-Scale HOG (460-D)   |
|   - Swin-T (Stage 3)          |                                   |   - Uniform LBP (270-D)       |
|   Output: 768-D Deep Vector   |                                   |   - Gabor Filter Bank (288-D) |
+---------------+---------------+                                   |   - SIFT-BoVW TF-IDF (208-D)  |
                |                                                   +---------------+---------------+
                |                                                                   |
                +-------------------------+   +-------------------------------------+
                                          |   |
                                          v   v
                      +---------------------------------------+
                      |        Cross-Modal Fusion Heads       |
                      |  1. Projected L2-Norm Concatenation   |
                      |  2. Gated Cross-Modal Attention       |
                      |  3. Token Self-Attention + SE Gate    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |          Clinical Classifier          |
                      |       (Normal vs. Pneumonia)          |
                      +---------------------------------------+
```

---

## Key Highlights

- **Overcoming Deep Texture Bias:** Standard deep CNNs and Vision Transformers can exhibit texture bias or oversmooth fine interstitial alveolar details. Handcrafted descriptors explicitly capture edge orientation distributions, micro-texture gradients, and frequency-domain energy.
- **Modern Vision Backbones:**
  - **ConvNeXt-Tiny:** Pure convolutional architecture modernizing 7x7 depthwise convolutions, inverted bottlenecks, and LayerNorm.
  - **Swin Transformer-T:** Hierarchical Vision Transformer utilizing shifted local windows with linear computational complexity.
- **Synchronized Augmentations:** Handcrafted features are precomputed on both standard and horizontally mirrored representations, ensuring synchronized data augmentation during deep training without feature correspondence degradation.
- **Multiple Fusion Paradigms:**
  - **Projected Concatenation:** Symmetric L2-normalized projection onto a shared 256-D manifold with a deep residual skip connection.
  - **Gated Cross-Modal Fusion:** Sigmoid gating network that balances handcrafted textural cues with bilinear interactions between deep and traditional features.
  - **Cross-Modal Token Attention:** 4-head multi-head self-attention treating deep and handcrafted embeddings as modality tokens, coupled with Squeeze-and-Excitation (SE) channel recalibration.

---

## Experimental Matrix

| Experiment ID | Backbone | Handcrafted Descriptor | Dimensionality | Fusion Strategy |
| :--- | :--- | :--- | :---: | :--- |
| `exp1_convnext_baseline` | ConvNeXt-Tiny | None (Pure Deep) | 0-D | Pure Deep Head |
| `exp2_convnext_hog` | ConvNeXt-Tiny | Multi-Scale HOG | 460-D | Projected L2-Norm Concatenation |
| `exp3_convnext_lbp` | ConvNeXt-Tiny | Multi-Scale Uniform LBP | 270-D | Gated Cross-Modal Interaction |
| `exp4_swin_baseline` | Swin-T | None (Pure Deep) | 0-D | Pure Deep Head |
| `exp5_swin_gabor` | Swin-T | 24-Kernel Gabor Bank | 288-D | Projected L2-Norm Concatenation |
| `exp6_swin_sift` | Swin-T | SIFT-BoVW with TF-IDF | 208-D | Token Cross-Attention + SE Gate |

---

## Handcrafted Feature Extractors

### 1. Multi-Scale Histogram of Oriented Gradients (HOG) — 460-D
- **Coarse Scale:** 9 orientation bins, $32 \times 32$ cell size, $2 \times 2$ block size ($36\text{-D}$).
- **Fine Scale:** 8 orientation bins, $16 \times 16$ cell size, pooled down across $4 \times 4$ spatial quadrants ($416\text{-D}$).
- **Gradient Statistics:** Global and quadrant Sobel magnitude percentiles and directional orientation means ($8\text{-D}$).

### 2. Multi-Scale Uniform Local Binary Patterns (LBP) — 270-D
- Extracts uniform circular LBP across three radii: $(P=8, R=1)$, $(P=16, R=2)$, and $(P=24, R=3)$.
- Computed across global radiographs and $2 \times 2$ spatial quadrants to capture both macro-texture and localized alveolar consolidations.

### 3. 24-Kernel Gabor Filter Bank — 288-D
- Decomposes radiographs across 4 spatial wavelengths ($\lambda \in \{4.0, 8.0, 12.0, 16.0\}$) and 6 orientations ($\theta \in \{0, \frac{\pi}{6}, \frac{\pi}{3}, \frac{\pi}{2}, \frac{2\pi}{3}, \frac{5\pi}{6}\}$).
- Captures multi-directional frequency response energy, standard deviations, and quadrant statistics.

### 4. SIFT Bag-of-Visual-Words (BoVW) with TF-IDF — 208-D
- Combines Difference-of-Gaussian (DoG) keypoints with dense grid sampling ($16\text{px}$ strides).
- RootSIFT normalization ($L_1 \to \sqrt{\cdot}$) clustered into a 128-word visual vocabulary via MiniBatch K-Means fitted strictly on training data.
- Normalized with term frequency-inverse document frequency ($\text{TF-IDF}$) and spatial quadrant histograms.

---

## Repository Structure

```
hybrid-radiograph-feature-fusion/
├── src/
│   ├── __init__.py               # Package initialization
│   ├── preprocessing.py          # Symmetric CLAHE equalization
│   ├── handcrafted_features.py   # HOG, LBP, Gabor, SIFT-BoVW extractors
│   ├── fusion_modules.py         # Projected Concat, Gated, Token Attention
│   ├── models.py                 # HybridMedicalClassifier (ConvNeXt & Swin-T)
│   ├── dataset.py                # Synchronized multi-modal PyTorch dataset
│   └── metrics.py                # Clinical metrics & ROC computation
├── prepare_dataset.py            # Downloads Kermany dataset & creates stratified splits
├── train.py                      # Master training runner across all 6 configurations
├── generate_figures.py           # Publication-quality figure generation (300 DPI)
├── requirements.txt              # Environment dependencies
├── .gitignore                    # Artifacts & data exclusions
└── README.md                     # Documentation
```

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Subhans1501/hybrid-radiograph-feature-fusion.git
cd hybrid-radiograph-feature-fusion
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download Kermany chest radiograph shards and build the 1,500-sample balanced stratified split (70% Train, 15% Val, 15% Test):
```bash
python prepare_dataset.py
```

### 3. Training & Benchmarking
Run all six hybrid deep and handcrafted feature fusion experiments with mixed precision:
```bash
python train.py
```
Outputs model weights to `outputs/checkpoints/` and performance statistics to `outputs/results.json`.

### 4. Generate Visualizations
Generate 300 DPI publication figures (ROC curves, confusion matrices, ablation comparisons, feature activations):
```bash
python generate_figures.py
```
Outputs high-resolution 300 DPI figures to `outputs/figures/`.

---

## Benchmark Results (Held-Out Test Set, n = 225)

| Experiment | Accuracy (%) | Precision (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ConvNeXt-Tiny` (Pure Deep Baseline) | **94.22** | **98.41** | 91.85 | **97.78** | **95.02** | 0.9878 |
| `ConvNeXt-Tiny` + Multi-Scale HOG (Projected Concat) | 93.33 | 96.15 | 92.59 | 94.44 | 94.34 | **0.9885** |
| `ConvNeXt-Tiny` + Uniform LBP (Gated Cross-Modal) | 93.78 | 96.90 | 92.59 | 95.56 | 94.70 | 0.9869 |
| `Swin-T` (Pure Deep Baseline) | 89.78 | 95.16 | 87.41 | 93.33 | 91.12 | 0.9843 |
| `Swin-T` + 24-Kernel Gabor Bank (Projected Concat) | 92.89 | 94.07 | **94.07** | 91.11 | 94.07 | 0.9878 |
| `Swin-T` + SIFT-BoVW TF-IDF (Token Attention + SE) | **93.78** | 96.18 | 93.33 | 94.44 | **94.74** | **0.9887** |

- Fusing **SIFT-BoVW TF-IDF** with **Swin-T** via Cross-Modal Token Self-Attention + SE boosted test accuracy by **+4.00%** (`89.78%` $\to$ `93.78%`), sensitivity by **+5.92%** (`87.41%` $\to$ `93.33%`), and achieved the highest overall **AUROC (0.9887)**.
- Fusing **24-Kernel Gabor Filter Bank** with **Swin-T** achieved the highest clinical sensitivity (**94.07%**).

### Visual Analysis

![ROC Curves](outputs/figures/fig5_roc_curves_comparison.png)
![Confusion Matrices](outputs/figures/fig4_confusion_matrices_all.png)
![t-SNE and Fusion Weights](outputs/figures/fig7_interpretability_tsne_and_gates.png)

---

## Clinical Metrics Evaluated
Every model is evaluated across comprehensive clinical diagnostics:
- **Accuracy:** Overall correctly classified radiographs.
- **Sensitivity / Recall:** Proportion of actual pneumonia cases detected ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). Critical for minimizing missed pathology.
- **Specificity:** Proportion of healthy controls correctly identified ($\frac{\text{TN}}{\text{TN} + \text{FP}}$).
- **Precision:** Positive predictive value ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
- **F1-Score & Macro-F1:** Harmonic mean of precision and recall.
- **AUROC:** Area Under the Receiver Operating Characteristic curve.

---

## License
This project is open-source under the [MIT License](LICENSE).
