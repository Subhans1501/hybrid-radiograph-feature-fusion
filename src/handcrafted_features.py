import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class HOGExtractor:
    """Multi-scale HOG + Sobel gradient statistics (460-D)."""

    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_single(self, img_gray: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        hog_coarse = hog(
            img, orientations=9, pixels_per_cell=(32, 32),
            cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True,
        )

        # Pool 8x8 fine cells down to 4x4 spatial bins to keep dimensionality compact
        hog_fine_map = hog(
            img, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), block_norm="L2-Hys", feature_vector=False,
        ).reshape(8, 8, 8)
        hog_fine = hog_fine_map.reshape(4, 2, 4, 2, 8).mean(axis=(1, 3)).flatten()

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.arctan2(gy, gx)
        grad_stats = np.array([
            mag.mean(), mag.std(), np.percentile(mag, 75), np.percentile(mag, 90),
            np.mean(np.abs(gx)), np.mean(np.abs(gy)),
            np.mean(np.cos(2 * ang)), np.mean(np.sin(2 * ang)),
        ], dtype=np.float32)

        return np.concatenate([hog_coarse, hog_fine, grad_stats]).astype(np.float32)

    def fit_transform(self, images: list) -> np.ndarray:
        raw = np.stack([self.extract_single(im) for im in images], axis=0)
        self.is_fitted = True
        return self.scaler.fit_transform(raw).astype(np.float32)

    def transform(self, images: list) -> np.ndarray:
        raw = np.stack([self.extract_single(im) for im in images], axis=0)
        return self.scaler.transform(raw).astype(np.float32)


class MultiScaleLBPExtractor:
    """Multi-radius uniform LBP over global + 2x2 spatial quadrants (270-D)."""

    def __init__(self, img_size: int = 128, scales=((8, 1), (16, 2), (24, 3))):
        self.img_size = img_size
        self.scales = scales
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def _pyramid_hist(lbp_map: np.ndarray, n_bins: int) -> np.ndarray:
        h, w = lbp_map.shape
        mh, mw = h // 2, w // 2
        regions = [
            lbp_map,
            lbp_map[:mh, :mw], lbp_map[:mh, mw:],
            lbp_map[mh:, :mw], lbp_map[mh:, mw:],
        ]
        hists = []
        for reg in regions:
            hist, _ = np.histogram(reg.ravel(), bins=n_bins, range=(0, n_bins), density=False)
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-6)
            hists.append(hist)
        return np.concatenate(hists)

    def extract_single(self, img_gray: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        feats = []
        for P, R in self.scales:
            lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
            feats.append(self._pyramid_hist(lbp, P + 2))
        return np.concatenate(feats).astype(np.float32)

    def fit_transform(self, images: list) -> np.ndarray:
        raw = np.stack([self.extract_single(im) for im in images], axis=0)
        self.is_fitted = True
        return self.scaler.fit_transform(raw).astype(np.float32)

    def transform(self, images: list) -> np.ndarray:
        raw = np.stack([self.extract_single(im) for im in images], axis=0)
        return self.scaler.transform(raw).astype(np.float32)


class GaborFilterBankExtractor:
    """24-kernel (4 scales x 6 orientations) Gabor filter bank with quadrant stats (288-D)."""

    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        self.wavelengths = [4.0, 8.0, 12.0, 16.0]
        self.orientations = [i * np.pi / 6.0 for i in range(6)]
        self.kernels = self._build_kernels()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _build_kernels(self):
        kernels = []
        for lambd in self.wavelengths:
            sigma = 0.56 * lambd
            ksize = int(max(7, round(sigma * 4) | 1))
            for theta in self.orientations:
                kern = cv2.getGaborKernel(
                    (ksize, ksize), sigma=sigma, theta=theta,
                    lambd=lambd, gamma=0.5, psi=0.0, ktype=cv2.CV_32F,
                )
                kern -= kern.mean()
                kern /= (np.linalg.norm(kern) + 1e-6)
                kernels.append(kern)
        return kernels

    def extract_single(self, img_gray: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mh, mw = img.shape[0] // 2, img.shape[1] // 2
        feats = []
        for kern in self.kernels:
            resp = np.abs(cv2.filter2D(img, cv2.CV_32F, kern))
            feats.extend([resp.mean(), resp.std(), np.sqrt(np.mean(resp ** 2)), np.percentile(resp, 90)])
            for q in (resp[:mh, :mw], resp[:mh, mw:], resp[mh:, :mw], resp[mh:, mw:]):
                feats.extend([q.mean(), q.std()])
        return np.array(feats, dtype=np.float32)

    def fit_transform(self, images: list) -> np.ndarray:
        raw = np.stack([self.extract_single(im) for im in images], axis=0)
        self.is_fitted = True
        return self.scaler.fit_transform(raw).astype(np.float32)

    def transform(self, images: list) -> np.ndarray:
        raw = np.stack([self.extract_single(im) for im in images], axis=0)
        return self.scaler.transform(raw).astype(np.float32)


class SIFTBoVWExtractor:
    """RootSIFT + train-only K-Means BoVW TF-IDF + spatial pyramid stats (208-D)."""

    def __init__(self, img_size: int = 128, n_clusters: int = 128, seed: int = 42):
        self.img_size = img_size
        self.n_clusters = n_clusters
        self.seed = seed
        self.sift = cv2.SIFT_create(nfeatures=200)
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, random_state=seed, n_init=3)
        self.idf = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Dense grid ensures homogeneous lung regions still produce descriptors
        self.dense_kps = []
        for y in range(16, img_size - 8, 16):
            for x in range(16, img_size - 8, 16):
                self.dense_kps.append(cv2.KeyPoint(float(x), float(y), 16.0))
                self.dense_kps.append(cv2.KeyPoint(float(x), float(y), 24.0))

    def _extract_sift_raw(self, img_gray: np.ndarray):
        img = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        dog_kps, dog_desc = self.sift.detectAndCompute(img, None)
        _, dense_desc = self.sift.compute(img, self.dense_kps)

        half = self.img_size / 2.0
        q_counts = np.zeros(4, dtype=np.float32)
        q_resp = [[] for _ in range(4)]
        q_size = [[] for _ in range(4)]
        for kp in (dog_kps or []):
            qi = (0 if kp.pt[1] < half else 2) + (0 if kp.pt[0] < half else 1)
            q_counts[qi] += 1.0
            q_resp[qi].append(kp.response)
            q_size[qi].append(kp.size)

        dog_stats = []
        for qi in range(4):
            dog_stats.extend([
                q_counts[qi] / 50.0,
                float(np.mean(q_resp[qi])) if q_resp[qi] else 0.0,
                float(np.mean(q_size[qi])) if q_size[qi] else 0.0,
                float(np.std(q_size[qi])) if len(q_size[qi]) > 1 else 0.0,
            ])
        dog_stats = np.array(dog_stats, dtype=np.float32)

        descs, quads = [], []
        if dog_desc is not None and len(dog_desc) > 0:
            descs.append(dog_desc)
            quads.extend([(0 if kp.pt[1] < half else 2) + (0 if kp.pt[0] < half else 1) for kp in dog_kps])
        if dense_desc is not None and len(dense_desc) > 0:
            descs.append(dense_desc)
            quads.extend([(0 if kp.pt[1] < half else 2) + (0 if kp.pt[0] < half else 1) for kp in self.dense_kps])

        # RootSIFT normalization (L1 norm followed by element-wise sqrt)
        all_desc = np.vstack(descs).astype(np.float32)
        all_desc /= (all_desc.sum(axis=1, keepdims=True) + 1e-7)
        all_desc = np.sqrt(np.maximum(all_desc, 0.0))
        return all_desc, np.array(quads, dtype=np.int32), dog_stats

    def _encode_from_raw(self, raw_list) -> np.ndarray:
        encoded = []
        n_coarse = 16
        group_size = self.n_clusters // n_coarse
        for desc, quads, dog_stats in raw_list:
            words = self.kmeans.predict(desc)
            tf = np.bincount(words, minlength=self.n_clusters).astype(np.float32)
            tf /= (tf.sum() + 1e-6)
            tfidf = tf * self.idf
            tfidf /= (np.linalg.norm(tfidf) + 1e-6)

            sp_hists = []
            for qi in range(4):
                w_q = words[quads == qi]
                if len(w_q) > 0:
                    h_q = np.bincount(w_q // group_size, minlength=n_coarse).astype(np.float32)
                    h_q /= (h_q.sum() + 1e-6)
                else:
                    h_q = np.zeros(n_coarse, dtype=np.float32)
                sp_hists.append(h_q)

            encoded.append(np.concatenate([tfidf, np.concatenate(sp_hists), dog_stats]).astype(np.float32))
        return np.stack(encoded, axis=0)

    def fit_transform(self, images: list) -> np.ndarray:
        raw_list = [self._extract_sift_raw(im) for im in images]
        rng = np.random.RandomState(self.seed)
        sampled = [
            desc[rng.choice(len(desc), size=80, replace=False)] if len(desc) > 80 else desc
            for desc, _, _ in raw_list
        ]
        self.kmeans.fit(np.vstack(sampled))

        df = np.zeros(self.n_clusters, dtype=np.float32)
        for desc, _, _ in raw_list:
            df[np.unique(self.kmeans.predict(desc))] += 1.0
        self.idf = np.log((len(raw_list) + 1.0) / (df + 1.0)) + 1.0

        raw_feats = self._encode_from_raw(raw_list)
        self.is_fitted = True
        return self.scaler.fit_transform(raw_feats).astype(np.float32)

    def transform(self, images: list) -> np.ndarray:
        raw_list = [self._extract_sift_raw(im) for im in images]
        return self.scaler.transform(self._encode_from_raw(raw_list)).astype(np.float32)
