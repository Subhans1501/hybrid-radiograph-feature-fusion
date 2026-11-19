import os
import sys
import time
import json
import copy
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessing import SymmetricCLAHE
from src.handcrafted_features import (
    HOGExtractor,
    MultiScaleLBPExtractor,
    GaborFilterBankExtractor,
    SIFTBoVWExtractor,
)
from src.dataset import HybridChestXRayDataset
from src.models import HybridMedicalClassifier
from src.metrics import compute_clinical_metrics

CONFIG = {
    "seed": 42,
    "img_size_deep": 224,
    "img_size_handcrafted": 128,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": [8, 8],
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "AdamW",
    "backbone_lr": 1e-4,
    "head_lr": 5e-4,
    "weight_decay": 1e-2,
    "scheduler": "CosineAnnealingLR",
    "label_smoothing": 0.05,
    "embed_dim": 256,
    "dropout": 0.30,
    "grad_clip_norm": 1.0,
}

# Check data paths
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")

MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clahe_splits():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Dataset manifest not found at {MANIFEST_PATH}. "
            "Please run 'python prepare_dataset.py' first."
        )
    df = pd.read_csv(MANIFEST_PATH)
    clahe = SymmetricCLAHE(CONFIG["clahe_clip_limit"], tuple(CONFIG["clahe_tile_grid"]))

    splits = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp].reset_index(drop=True)
        imgs_orig, imgs_flip = [], []
        for rel_p in sub["rel_path"]:
            gray = cv2.imread(os.path.join(DATA_DIR, rel_p), cv2.IMREAD_GRAYSCALE)
            eq = clahe(gray)
            imgs_orig.append(eq)
            imgs_flip.append(cv2.flip(eq, 1))
        splits[sp] = {
            "df": sub,
            "imgs": imgs_orig,
            "imgs_flip": imgs_flip,
            "labels": sub["label"].values.astype(np.int64),
        }
    return splits


def precompute_handcrafted_features(splits):
    extractors = {
        "hog": HOGExtractor(img_size=CONFIG["img_size_handcrafted"]),
        "lbp": MultiScaleLBPExtractor(img_size=CONFIG["img_size_handcrafted"]),
        "gabor": GaborFilterBankExtractor(img_size=CONFIG["img_size_handcrafted"]),
        "sift_bovw": SIFTBoVWExtractor(img_size=CONFIG["img_size_handcrafted"], n_clusters=128, seed=CONFIG["seed"]),
    }

    features = {}
    print("Precomputing handcrafted descriptors...")
    for name, ext in extractors.items():
        t0 = time.time()
        tr_orig = ext.fit_transform(splits["train"]["imgs"])
        tr_flip = ext.transform(splits["train"]["imgs_flip"])
        va_orig = ext.transform(splits["val"]["imgs"])
        te_orig = ext.transform(splits["test"]["imgs"])
        dt = time.time() - t0
        features[name] = {
            "train_orig": tr_orig,
            "train_flip": tr_flip,
            "val": va_orig,
            "test": te_orig,
            "dim": int(tr_orig.shape[1]),
            "extract_time_sec": round(dt, 2),
        }
        print(f"  {name:10s} -> feature dim={tr_orig.shape[1]:3d} ({dt:.1f}s)")
    return features


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, collect_artifacts=False):
    model.eval()
    total_loss, n_samples = 0.0, 0
    y_true, y_pred, y_prob = [], [], []
    all_embs, all_attn = [], []

    for imgs, h_feats, labels in loader:
        imgs, h_feats, labels = imgs.to(device), h_feats.to(device), labels.to(device)
        logits, emb, attn_info = model(imgs, h_feats)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        y_prob.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())

        if collect_artifacts:
            all_embs.append(emb.cpu().numpy())
            if attn_info is not None:
                all_attn.append(attn_info.cpu().numpy())

    metrics = compute_clinical_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = round(float(total_loss / max(1, n_samples)), 4)

    artifacts = None
    if collect_artifacts:
        artifacts = {
            "y_true": np.array(y_true, dtype=np.int64),
            "y_pred": np.array(y_pred, dtype=np.int64),
            "y_prob": np.array(y_prob, dtype=np.float32),
            "embeddings": np.concatenate(all_embs, axis=0) if all_embs else None,
            "attn_weights": np.concatenate(all_attn, axis=0) if all_attn else None,
        }
    return metrics, artifacts


def run_single_experiment(exp_spec, splits, hand_features, device):
    exp_id = exp_spec["id"]
    print(f"\n=======================================================")
    print(f"Running [{exp_id}]: {exp_spec['title']}")
    print(f"=======================================================")
    seed_everything(CONFIG["seed"])

    feat_key = exp_spec["feature"]
    if feat_key == "none":
        hand_dim, tr_h_o, tr_h_f, va_h, te_h = 0, None, None, None, None
    else:
        fdict = hand_features[feat_key]
        hand_dim = fdict["dim"]
        tr_h_o, tr_h_f, va_h, te_h = fdict["train_orig"], fdict["train_flip"], fdict["val"], fdict["test"]

    train_ds = HybridChestXRayDataset(splits["train"]["imgs"], splits["train"]["labels"], tr_h_o, tr_h_f, True, CONFIG["img_size_deep"])
    val_ds = HybridChestXRayDataset(splits["val"]["imgs"], splits["val"]["labels"], va_h, None, False, CONFIG["img_size_deep"])
    test_ds = HybridChestXRayDataset(splits["test"]["imgs"], splits["test"]["labels"], te_h, None, False, CONFIG["img_size_deep"])

    g = torch.Generator().manual_seed(CONFIG["seed"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = HybridMedicalClassifier(
        backbone_name=exp_spec["backbone"],
        fusion_type=exp_spec["fusion"],
        hand_dim=hand_dim,
        embed_dim=CONFIG["embed_dim"],
        num_classes=2,
        dropout=CONFIG["dropout"],
        pretrained=True,
    ).to(device)
    total_params, trainable_params = model.count_parameters()

    tr_labels = splits["train"]["labels"]
    n_norm, n_pneu = float((tr_labels == 0).sum()), float((tr_labels == 1).sum())
    n_tot = n_norm + n_pneu
    class_weights = torch.tensor([n_tot / (2.0 * n_norm), n_tot / (2.0 * n_pneu)], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.get_param_groups(CONFIG["backbone_lr"], CONFIG["head_lr"], CONFIG["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1_macro": [], "val_auroc": []}
    best_score, best_epoch, best_state = -1.0, 0, None
    t_start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        ep_t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, h_feats, labels in train_loader:
            imgs, h_feats, labels = imgs.to(device), h_feats.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits, _, _ = model(imgs, h_feats)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _, _ = model(imgs, h_feats)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
                optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += bs

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0
        val_metrics, _ = evaluate_model(model, val_loader, criterion, device, collect_artifacts=False)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_auroc"].append(val_metrics["auroc"])

        val_composite = 0.5 * val_metrics["f1_macro"] + 0.5 * (val_metrics["auroc"] * 100.0)
        if val_composite > best_score:
            best_score, best_epoch = val_composite, epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Ep {epoch:02d}/{CONFIG['epochs']:02d} ({time.time()-ep_t0:4.1f}s) | "
              f"TrAcc {train_acc:5.2f}% | ValAcc {val_metrics['accuracy']:5.2f}% | ValAUC {val_metrics['auroc']:.4f}")

    total_train_sec = round(time.time() - t_start, 1)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(CKPT_DIR, f"{exp_id}.pth"))

    val_best_metrics, _ = evaluate_model(model, val_loader, criterion, device, collect_artifacts=False)
    test_metrics, test_artifacts = evaluate_model(model, test_loader, criterion, device, collect_artifacts=True)

    attn_summary = None
    if test_artifacts["attn_weights"] is not None:
        aw, yt = test_artifacts["attn_weights"], test_artifacts["y_true"]
        attn_summary = {
            "overall_mean": round(float(aw.mean()), 4),
            "overall_std": round(float(aw.std()), 4),
            "normal_mean": round(float(aw[yt == 0].mean()), 4),
            "pneumonia_mean": round(float(aw[yt == 1].mean()), 4),
        }

    print(f"  --> Test [{exp_id}] | Acc: {test_metrics['accuracy']:.2f}% | F1: {test_metrics['f1_score']:.2f}% | AUROC: {test_metrics['auroc']:.4f}")

    return {
        "id": exp_id,
        "title": exp_spec["title"],
        "backbone": exp_spec["backbone"],
        "feature_extractor": exp_spec["feature_label"],
        "feature_key": exp_spec["feature"],
        "feature_dim": hand_dim,
        "fusion_strategy": exp_spec["fusion_label"],
        "fusion_key": exp_spec["fusion"],
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "best_epoch": int(best_epoch),
        "train_time_sec": total_train_sec,
        "val_metrics": val_best_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "attn_summary": attn_summary,
    }, test_artifacts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation Device: {device} (PyTorch {torch.__version__})")

    splits = load_clahe_splits()
    hand_features = precompute_handcrafted_features(splits)

    experiments = [
        {
            "id": "exp1_convnext_baseline",
            "title": "ConvNeXt-Tiny (Pure Deep Baseline)",
            "backbone": "convnext_tiny",
            "feature": "none",
            "feature_label": "None (Pure Deep Features)",
            "fusion": "none",
            "fusion_label": "None (Baseline Head)",
        },
        {
            "id": "exp2_convnext_hog",
            "title": "ConvNeXt-Tiny + Multi-Scale HOG (Projected Concat)",
            "backbone": "convnext_tiny",
            "feature": "hog",
            "feature_label": "Multi-Scale Spatial HOG (460-D)",
            "fusion": "projected_concat",
            "fusion_label": "Projected Dimensionality-Aligned Concatenation",
        },
        {
            "id": "exp3_convnext_lbp",
            "title": "ConvNeXt-Tiny + Multi-Scale Uniform LBP (Gated Cross-Modal)",
            "backbone": "convnext_tiny",
            "feature": "lbp",
            "feature_label": "Multi-Scale Pyramid Uniform LBP (270-D)",
            "fusion": "gated_attention",
            "fusion_label": "Gated Cross-Modal Attention Fusion",
        },
        {
            "id": "exp4_swin_baseline",
            "title": "Swin-T (Pure Deep Baseline)",
            "backbone": "swin_t",
            "feature": "none",
            "feature_label": "None (Pure Deep Features)",
            "fusion": "none",
            "fusion_label": "None (Baseline Head)",
        },
        {
            "id": "exp5_swin_gabor",
            "title": "Swin-T + 24-Kernel Gabor Filter Bank (Projected Concat)",
            "backbone": "swin_t",
            "feature": "gabor",
            "feature_label": "24-Kernel Gabor Filter Bank (288-D)",
            "fusion": "projected_concat",
            "fusion_label": "Projected Dimensionality-Aligned Concatenation",
        },
        {
            "id": "exp6_swin_sift",
            "title": "Swin-T + SIFT-BoVW TF-IDF (Token Attention + SE)",
            "backbone": "swin_t",
            "feature": "sift_bovw",
            "feature_label": "SIFT + K-Means BoVW TF-IDF (208-D)",
            "fusion": "token_attention",
            "fusion_label": "Multi-Head Cross-Modal Token Attention + SE",
        },
    ]

    all_results, artifact_bundle = [], {}
    for spec in experiments:
        res, art = run_single_experiment(spec, splits, hand_features, device)
        all_results.append(res)
        artifact_bundle[f"{spec['id']}_y_true"] = art["y_true"]
        artifact_bundle[f"{spec['id']}_y_pred"] = art["y_pred"]
        artifact_bundle[f"{spec['id']}_y_prob"] = art["y_prob"]
        if art["embeddings"] is not None:
            artifact_bundle[f"{spec['id']}_emb"] = art["embeddings"]
        if art["attn_weights"] is not None:
            artifact_bundle[f"{spec['id']}_attn"] = art["attn_weights"]

        payload = {
            "config": CONFIG,
            "dataset_splits": {
                sp: {
                    "total": int(len(splits[sp]["labels"])),
                    "normal": int((splits[sp]["labels"] == 0).sum()),
                    "pneumonia": int((splits[sp]["labels"] == 1).sum()),
                }
                for sp in ["train", "val", "test"]
            },
            "feature_dims": {k: v["dim"] for k, v in hand_features.items()},
            "experiments": all_results,
        }
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(payload, f, indent=4)

    np.savez_compressed(os.path.join(OUT_DIR, "eval_artifacts.npz"), **artifact_bundle)
    print("\nBenchmark completed. Summary saved to outputs/results.json and eval_artifacts.npz")


if __name__ == "__main__":
    main()
