import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import hog, local_binary_pattern
from sklearn.manifold import TSNE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessing import SymmetricCLAHE

DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")
ARTIFACTS_PATH = os.path.join(OUT_DIR, "eval_artifacts.npz")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def make_fig1_preprocessing_and_handcrafted():
    manifest_p = os.path.join(DATA_DIR, "manifest.csv")
    if not os.path.exists(manifest_p):
        print(f"Manifest not found at {manifest_p}. Skipping Fig 1.")
        return
    df = pd.read_csv(manifest_p)
    test_df = df[df["split"] == "test"]
    norm_row = test_df[test_df["label"] == 0].iloc[2]
    pneu_row = test_df[test_df["label"] == 1].iloc[2]

    clahe = SymmetricCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    sift = cv2.SIFT_create(nfeatures=120)
    gabor_kernels = [
        cv2.getGaborKernel((15, 15), 4.0, th, 8.0, 0.5, 0.0, ktype=cv2.CV_32F)
        for th in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    ]

    fig, axes = plt.subplots(2, 6, figsize=(16, 5.6))
    col_titles = [
        "1. Raw Radiograph",
        "2. Symmetric CLAHE\nEnhancement",
        "3. HOG Gradient\nOrientation Map",
        "4. Multi-Scale\nUniform LBP (R=2)",
        "5. Gabor Filter Bank\nEnergy Response",
        "6. SIFT Keypoints\n(Salient Descriptors)",
    ]

    for row_i, (row_label, row) in enumerate([("NORMAL\n(Healthy)", norm_row), ("PNEUMONIA\n(Infiltrate)", pneu_row)]):
        raw = cv2.imread(os.path.join(DATA_DIR, row["rel_path"]), cv2.IMREAD_GRAYSCALE)
        eq = clahe(raw)
        eq128 = cv2.resize(eq, (128, 128), interpolation=cv2.INTER_AREA)

        _, hog_vis = hog(
            eq128, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys"
        )
        lbp_map = local_binary_pattern(eq128, P=16, R=2, method="uniform")
        f_img = eq128.astype(np.float32) / 255.0
        g_resp = np.mean([np.abs(cv2.filter2D(f_img, cv2.CV_32F, k)) for k in gabor_kernels], axis=0)
        kps, _ = sift.detectAndCompute(eq128, None)
        sift_vis = cv2.drawKeypoints(
            eq128, kps[:80], None, color=(255, 80, 40),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        panels = [(raw, "gray"), (eq, "gray"), (hog_vis, "inferno"),
                  (lbp_map, "viridis"), (g_resp, "magma"), (sift_vis, None)]
        for col_i, (img_show, cmap) in enumerate(panels):
            ax = axes[row_i, col_i]
            ax.imshow(img_show, cmap=cmap) if cmap else ax.imshow(img_show)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9.5, pad=6)
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold", labelpad=8)

    fig.suptitle("Symmetric CLAHE and Four Multi-Scale Handcrafted Feature Extractors",
                 fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_preprocessing_handcrafted_pipeline.png"))
    plt.close()


def make_fig2_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 6.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#1B365D", tc="black", fs=8.5, bold=False):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.4, shrinkA=2, shrinkB=2))

    ax.text(7.0, 6.12, "Hybrid Deep + Handcrafted Feature-Fusion Architectures",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#102C57")

    box(0.25, 2.35, 2.15, 1.5,
        "Chest X-Ray Input\n(Kermany Cohort)\n\nSymmetric CLAHE\n(clip=2.0, 8x8 tiles)",
        fc="#EBF5FB", ec="#1F618D", bold=True, fs=8.5)

    # ConvNeXt branch
    ax.add_patch(patches.FancyBboxPatch((2.7, 3.25), 11.05, 2.55, boxstyle="round,pad=0.05",
                                        fc="#F8FAFC", ec="#2874A6", ls="--", lw=1.2))
    ax.text(8.2, 5.62, "Modern ConvNet Stream: ConvNeXt-Tiny Backbone",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1B4F72")
    box(2.95, 4.55, 2.55, 0.85, "Deep Stream (224x224)\nConvNeXt-Tiny (Stage 3 FT)\n768-D Feature Embedding", fc="#D6EAF8", ec="#1F618D")
    box(2.95, 3.45, 2.55, 0.85, "Handcrafted Stream (128x128)\nMulti-Scale HOG (460-D) /\nPyramid Uniform LBP (270-D)", fc="#FEF9E7", ec="#B7950B")
    box(6.05, 4.55, 3.45, 0.85, "Fusion A: Projected Concatenation\nL2-Norm 256-D + 256-D -> 512-D MLP\n+ Deep Residual Skip", fc="#E8F8F5", ec="#117A65")
    box(6.05, 3.45, 3.45, 0.85, "Fusion B: Gated Cross-Modal Attention\nalpha = sigmoid(W_g[f_d || f_h])\nf_d + alpha*f_h + (1-alpha)*tanh(f_d*f_h)", fc="#FDEDEC", ec="#943126")
    box(10.05, 4.00, 1.55, 0.85, "256-D Fused\nEmbedding\n(LayerNorm)", fc="#EBF5FB", ec="#1F618D", bold=True)
    box(11.95, 4.00, 1.55, 0.85, "Classifier\nNormal vs.\nPneumonia", fc="#102C57", ec="#102C57", tc="white", bold=True)

    for (x1, y1, x2, y2) in [(2.4, 3.4, 2.95, 4.95), (2.4, 3.2, 2.95, 3.85), (5.5, 4.95, 6.05, 4.95),
                              (5.5, 4.75, 6.05, 4.05), (5.5, 3.95, 6.05, 4.75), (5.5, 3.85, 6.05, 3.85),
                              (9.5, 4.95, 10.05, 4.55), (9.5, 3.85, 10.05, 4.25), (11.6, 4.42, 11.95, 4.42)]:
        arrow(x1, y1, x2, y2)

    # Swin Transformer branch
    ax.add_patch(patches.FancyBboxPatch((2.7, 0.35), 11.05, 2.55, boxstyle="round,pad=0.05",
                                        fc="#FAF9FC", ec="#6C3483", ls="--", lw=1.2))
    ax.text(8.2, 2.72, "Vision Transformer Stream: Swin Transformer-T Backbone",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#4A235A")
    box(2.95, 1.65, 2.55, 0.85, "Deep Stream (224x224)\nSwin-T (Stage 3 FT)\n768-D Feature Embedding", fc="#EBDEF0", ec="#6C3483")
    box(2.95, 0.55, 2.55, 0.85, "Handcrafted Stream (128x128)\n24-Kernel Gabor Bank (288-D) /\nSIFT-BoVW TF-IDF (208-D)", fc="#FEF9E7", ec="#B7950B")
    box(6.05, 1.65, 3.45, 0.85, "Fusion A: Projected Concatenation\nL2-Norm 256-D + 256-D -> 512-D MLP\n+ Deep Residual Skip", fc="#E8F8F5", ec="#117A65")
    box(6.05, 0.55, 3.45, 0.85, "Fusion C: Cross-Modal Token Attention\n4-Head MHSA over [t_deep, t_hand]\n+ Squeeze-and-Excitation (SE) Gate", fc="#FDEDEC", ec="#943126")
    box(10.05, 1.10, 1.55, 0.85, "256-D Fused\nEmbedding\n(LayerNorm)", fc="#EBDEF0", ec="#6C3483", bold=True)
    box(11.95, 1.10, 1.55, 0.85, "Classifier\nNormal vs.\nPneumonia", fc="#4A235A", ec="#4A235A", tc="white", bold=True)

    for (x1, y1, x2, y2) in [(2.4, 2.95, 2.95, 2.05), (2.4, 2.75, 2.95, 0.95), (5.5, 2.05, 6.05, 2.05),
                              (5.5, 1.85, 6.05, 1.15), (5.5, 1.05, 6.05, 1.85), (5.5, 0.95, 6.05, 0.95),
                              (9.5, 2.05, 10.05, 1.65), (9.5, 0.95, 10.05, 1.35), (11.6, 1.52, 11.95, 1.52)]:
        arrow(x1, y1, x2, y2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_architecture_fusion_diagram.png"))
    plt.close()


def make_fig3_training_curves(results):
    runs = results.get("experiments", results.get("runs", []))
    if not runs:
        return
    epochs = list(range(1, results["config"]["epochs"] + 1))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    colors = ["#2E4053", "#1F618D", "#C0392B"]
    markers = ["o", "s", "^"]
    short_c = ["ConvNeXt-Tiny (Base)", "+ HOG (Concat)", "+ LBP (Gated Attn)"]
    short_s = ["Swin-T (Base)", "+ Gabor (Concat)", "+ SIFT-BoVW (Token Attn)"]

    for idx, (r, lbl) in enumerate(zip(runs[:3], short_c)):
        axes[0, 0].plot(epochs, r["history"]["val_loss"], marker=markers[idx], color=colors[idx], lw=2, label=lbl)
        axes[0, 1].plot(epochs, r["history"]["val_auroc"], marker=markers[idx], color=colors[idx], lw=2, label=lbl)
    for idx, (r, lbl) in enumerate(zip(runs[3:], short_s)):
        axes[1, 0].plot(epochs, r["history"]["val_loss"], marker=markers[idx], color=colors[idx], lw=2, label=lbl)
        axes[1, 1].plot(epochs, r["history"]["val_auroc"], marker=markers[idx], color=colors[idx], lw=2, label=lbl)

    axes[0, 0].set_title("ConvNeXt-Tiny Family - Validation Loss")
    axes[0, 1].set_title("ConvNeXt-Tiny Family - Validation AUROC")
    axes[1, 0].set_title("Swin Transformer-T Family - Validation Loss")
    axes[1, 1].set_title("Swin Transformer-T Family - Validation AUROC")
    for r_i in range(2):
        axes[r_i, 0].set_ylabel("Weighted CE Loss")
        axes[r_i, 1].set_ylabel("Validation AUROC")
        for c_i in range(2):
            axes[r_i, c_i].set_xlabel("Epoch")
            axes[r_i, c_i].set_xticks(epochs)
            axes[r_i, c_i].grid(True, linestyle="--", alpha=0.45)
            axes[r_i, c_i].legend(fontsize=8.5, loc="best")

    fig.suptitle("Validation Convergence Across All Architecture Configurations", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_training_curves_all.png"))
    plt.close()


def make_fig4_confusion_matrices(results):
    runs = results.get("experiments", results.get("runs", []))
    if not runs:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    titles = [
        "ConvNeXt-Tiny\n(Baseline)",
        "ConvNeXt-Tiny + HOG\n(Projected Concat)",
        "ConvNeXt-Tiny + LBP\n(Gated Attention)",
        "Swin-T\n(Baseline)",
        "Swin-T + Gabor\n(Projected Concat)",
        "Swin-T + SIFT-BoVW\n(Token Attn + SE)",
    ]

    for idx, (r, title) in enumerate(zip(runs, titles)):
        ax = axes[idx // 3, idx % 3]
        cm_d = r["test_metrics"]["confusion_matrix"]
        mat = np.array([[cm_d["tn"], cm_d["fp"]], [cm_d["fn"], cm_d["tp"]]])
        pct = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1) * 100.0

        ax.imshow(mat, cmap="Blues" if idx < 3 else "Purples", vmin=0, vmax=mat.max() * 1.15)
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.55 else "black"
                ax.text(j, i, f"{val}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Normal", "Pred: Pneumonia"], fontsize=9)
        ax.set_yticklabels(["True: Normal", "True: Pneumonia"], fontsize=9)
        tm = r["test_metrics"]
        ax.set_title(f"{title}\nAcc: {tm['accuracy']:.2f}% | F1: {tm['f1_score']:.2f}%", fontsize=9.5)

    fig.suptitle("Held-Out Test Confusion Matrices (n = 225 Radiographs)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_confusion_matrices_all.png"))
    plt.close()


def make_fig5_roc_curves(results):
    runs = results.get("experiments", results.get("runs", []))
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    colors = ["#2E4053", "#1F618D", "#C0392B"]
    lss = ["--", "-", "-"]

    groups = [
        (axes[0], runs[:3], "ConvNeXt-Tiny Architecture Runs", ["Baseline", "+ HOG (Concat)", "+ LBP (Gated Attn)"]),
        (axes[1], runs[3:], "Swin-T Architecture Runs", ["Baseline", "+ Gabor (Concat)", "+ SIFT-BoVW (Token Attn)"]),
    ]

    for ax, grp, title, labels in groups:
        for i, (r, lbl) in enumerate(zip(grp, labels)):
            roc = r["test_metrics"]["roc_curve"]
            ax.plot(roc["fpr"], roc["tpr"], color=colors[i], linestyle=lss[i], lw=2.3,
                    label=f"{lbl} (AUROC = {r['test_metrics']['auroc']:.4f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1.2, alpha=0.6, label="Random Guess (0.5000)")
        ax.axhline(0.871, color="#27AE60", linestyle="-.", lw=1.3, alpha=0.75, label="Standard Benchmark AUROC = 0.8710")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.03)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8.8)

    fig.suptitle("Test Set ROC Curves Across All Architecture Configurations", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_roc_curves_comparison.png"))
    plt.close()


def make_fig6_grouped_metrics(results):
    runs = results.get("experiments", results.get("runs", []))
    if not runs:
        return
    labels = ["ConvNeXt\n(Base)", "ConvNeXt\n+HOG", "ConvNeXt\n+LBP",
              "Swin-T\n(Base)", "Swin-T\n+Gabor", "Swin-T\n+SIFT"]
    metrics_keys = [
        ("accuracy", "Accuracy", "#1B4F72"),
        ("precision", "Precision", "#2874A6"),
        ("recall_sensitivity", "Sensitivity", "#117A65"),
        ("specificity", "Specificity", "#B7950B"),
        ("f1_score", "F1-Score", "#A93226"),
    ]

    x = np.arange(len(runs))
    width = 0.155
    fig, ax = plt.subplots(figsize=(14, 5.6))

    for m_i, (key, name, color) in enumerate(metrics_keys):
        vals = [r["test_metrics"][key] for r in runs]
        bars = ax.bar(x + (m_i - 2) * width, vals, width, label=name, color=color, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2, rotation=90)

    ax.axvline(2.5, color="#7F8C8D", linestyle="--", lw=1.2)
    ax.text(1.0, 105.5, "ConvNeXt-Tiny Series", ha="center", fontsize=10, fontweight="bold", color="#1B4F72")
    ax.text(4.0, 105.5, "Swin-T Series", ha="center", fontsize=10, fontweight="bold", color="#4A235A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(70, 109)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=True)
    ax.set_title("Clinical Test Evaluation Metrics Across Architecture Configurations", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_grouped_metrics_comparison.png"))
    plt.close()


def make_fig7_interpretability(results):
    if not os.path.exists(ARTIFACTS_PATH):
        print(f"Artifacts file not found at {ARTIFACTS_PATH}. Skipping Fig 7.")
        return
    art = np.load(ARTIFACTS_PATH)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    keys = list(art.keys())
    c_emb_k = [k for k in keys if "lbp" in k and k.endswith("_emb")]
    s_emb_k = [k for k in keys if "sift" in k and k.endswith("_emb")]

    if c_emb_k and s_emb_k:
        c_k = c_emb_k[0].replace("_emb", "")
        s_k = s_emb_k[0].replace("_emb", "")

        for ax, key_prefix, title, cols in [
            (axes[0], c_k, "t-SNE: ConvNeXt-Tiny + LBP\n(Gated Attention)", ("#2874A6", "#C0392B")),
            (axes[1], s_k, "t-SNE: Swin-T + SIFT-BoVW\n(Token Attention + SE)", ("#6C3483", "#D35400")),
        ]:
            emb = art[f"{key_prefix}_emb"]
            y = art[f"{key_prefix}_y_true"]
            z = TSNE(n_components=2, perplexity=25, random_state=42).fit_transform(emb)
            for cls, name, col in [(0, "Normal", cols[0]), (1, "Pneumonia", cols[1])]:
                m = (y == cls)
                ax.scatter(z[m, 0], z[m, 1], c=col, label=name, alpha=0.78, s=28, edgecolors="white", lw=0.4)
            ax.set_title(title)
            ax.set_xlabel("t-SNE Dim 1")
            ax.set_ylabel("t-SNE Dim 2")
            ax.legend(fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.35)

        y1 = art[f"{c_k}_y_true"]
        y2 = art[f"{s_k}_y_true"]
        g_lbp = art[f"{c_k}_attn"]
        a_sift = art[f"{s_k}_attn"]

        bp = axes[2].boxplot(
            [g_lbp[y1 == 0], g_lbp[y1 == 1], a_sift[y2 == 0], a_sift[y2 == 1]],
            tick_labels=["ConvNeXt+LBP\nNormal", "ConvNeXt+LBP\nPneumonia", "Swin+SIFT\nNormal", "Swin+SIFT\nPneumonia"],
            patch_artist=True, widths=0.52,
        )
        for patch, c in zip(bp["boxes"], ["#AED6F1", "#F5B7B1", "#D7BDE2", "#FAD7A0"]):
            patch.set_facecolor(c)
            patch.set_edgecolor("#2C3E50")
        axes[2].set_title("Learned Fusion / Cross-Attention Weights\non Handcrafted Features (Test Set)")
        axes[2].set_ylabel("Mean Modality Weight")
        axes[2].tick_params(axis="x", labelsize=8)
        axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)

        fig.suptitle("Fused Latent Representation Separability and Adaptive Modality Weights", fontsize=12.5, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig7_interpretability_tsne_and_gates.png"))
        plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    make_fig1_preprocessing_and_handcrafted()
    make_fig2_architecture_diagram()

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)
        make_fig3_training_curves(results)
        make_fig4_confusion_matrices(results)
        make_fig5_roc_curves(results)
        make_fig6_grouped_metrics(results)
        make_fig7_interpretability(results)
    print(f"Figure generation completed. Figures stored in: {FIG_DIR}")


if __name__ == "__main__":
    main()
