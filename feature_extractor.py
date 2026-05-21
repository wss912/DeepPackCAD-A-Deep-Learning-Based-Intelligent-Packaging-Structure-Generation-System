# feature_extractor.py
import numpy as np

def extract_features(L, W, H):
    V = L * W * H
    return np.array([
        L, W, H,
        V,
        L / W,
        L / H,
        W / H
    ], dtype=np.float32)