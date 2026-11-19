import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HybridChestXRayDataset(Dataset):
    def __init__(self, clahe_images: list, labels: np.ndarray,
                 hand_feats_orig: np.ndarray = None, hand_feats_flip: np.ndarray = None,
                 is_train: bool = False, img_size: int = 224):
        self.clahe_images = clahe_images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.hand_feats_orig = hand_feats_orig
        self.hand_feats_flip = hand_feats_flip
        self.is_train = is_train

        if is_train:
            self.spatial_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=7, fill=0),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.spatial_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.clahe_images)

    def __getitem__(self, idx: int):
        img_gray = self.clahe_images[idx]
        flipped = False
        if self.is_train and random.random() < 0.5:
            img_gray = cv2.flip(img_gray, 1)
            flipped = True

        img_tensor = self.spatial_tf(Image.fromarray(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)))

        if self.hand_feats_orig is not None:
            h_vec = self.hand_feats_flip[idx] if (flipped and self.hand_feats_flip is not None) else self.hand_feats_orig[idx]
            hand_tensor = torch.from_numpy(h_vec).float()
        else:
            hand_tensor = torch.zeros(1, dtype=torch.float32)

        return img_tensor, hand_tensor, int(self.labels[idx])
