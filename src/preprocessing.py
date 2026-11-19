import cv2
import numpy as np


class SymmetricCLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_gray: np.ndarray) -> np.ndarray:
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        return self.clahe.apply(img_gray)
