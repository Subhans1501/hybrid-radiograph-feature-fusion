import io
import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_OUT = os.path.join(BASE_DIR, "data", "chest_xray")
IMG_DIR = os.path.join(DATA_OUT, "images")
MANIFEST_PATH = os.path.join(DATA_OUT, "manifest.csv")
STATS_PATH = os.path.join(DATA_OUT, "dataset_stats.json")

TARGET_NORMAL = 600
TARGET_PNEUMONIA = 900
SEED = 42


def decode_pil_image(img_cell) -> np.ndarray:
    if isinstance(img_cell, dict) and "bytes" in img_cell:
        pil = Image.open(io.BytesIO(img_cell["bytes"])).convert("L")
    elif isinstance(img_cell, Image.Image):
        pil = img_cell.convert("L")
    else:
        raise TypeError(f"Unexpected image entry type: {type(img_cell)}")
    return np.array(pil, dtype=np.uint8)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    parquet_files = [
        "data/train-00000-of-00007.parquet",
        "data/train-00002-of-00007.parquet",
        "data/validation-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
    ]

    dfs = []
    print("Fetching Kermany Chest X-Ray dataset shards from Hugging Face...")
    for fname in parquet_files:
        local_p = hf_hub_download(
            repo_id="hf-vision/chest-xray-pneumonia", filename=fname, repo_type="dataset"
        )
        df_part = pd.read_parquet(local_p)
        df_part["source_shard"] = os.path.basename(fname)
        dfs.append(df_part)

    df_all = pd.concat(dfs, ignore_index=True)
    df_norm = df_all[df_all["label"] == 0].sample(
        n=min(TARGET_NORMAL, (df_all["label"] == 0).sum()), random_state=SEED
    )
    df_pneu = df_all[df_all["label"] == 1].sample(
        n=min(TARGET_PNEUMONIA, (df_all["label"] == 1).sum()), random_state=SEED
    )
    df_sub = pd.concat([df_norm, df_pneu], ignore_index=True).sample(
        frac=1.0, random_state=SEED
    ).reset_index(drop=True)

    # Stratified split: 70% Train, 15% Validation, 15% Test
    y = df_sub["label"].values
    train_idx, temp_idx = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=SEED).split(np.zeros(len(y)), y))
    val_rel, test_rel = next(StratifiedShuffleSplit(1, test_size=0.50, random_state=SEED).split(np.zeros(len(temp_idx)), y[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    splits = np.array([""] * len(df_sub), dtype=object)
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"

    print("Writing images to disk...")
    records = []
    for idx, row in df_sub.iterrows():
        img_gray = decode_pil_image(row["image"])
        img_256 = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)

        label_int = int(row["label"])
        label_name = "PNEUMONIA" if label_int == 1 else "NORMAL"
        filename = f"{splits[idx]}_{label_name.lower()}_{idx:04d}.png"
        cv2.imwrite(os.path.join(IMG_DIR, filename), img_256)

        records.append({
            "id": idx,
            "filename": filename,
            "rel_path": os.path.join("images", filename),
            "label": label_int,
            "class_name": label_name,
            "split": splits[idx],
            "source_shard": row["source_shard"],
            "orig_h": int(img_gray.shape[0]),
            "orig_w": int(img_gray.shape[1]),
        })

    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_PATH, index=False)

    stats = {
        "total_images": int(len(manifest)),
        "seed": SEED,
        "splits": {
            sp: {
                "total": int((manifest["split"] == sp).sum()),
                "normal": int(((manifest["split"] == sp) & (manifest["label"] == 0)).sum()),
                "pneumonia": int(((manifest["split"] == sp) & (manifest["label"] == 1)).sum()),
            }
            for sp in ["train", "val", "test"]
        },
        "orig_height_range": [int(manifest["orig_h"].min()), int(manifest["orig_h"].max())],
        "orig_width_range": [int(manifest["orig_w"].min()), int(manifest["orig_w"].max())],
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
    print("Dataset prepared successfully:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
