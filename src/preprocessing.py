import os
import cv2
import numpy as np
from glob import glob

def load_image(path, target_size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def load_mask(path, target_size=(256, 256)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    mask = (mask > 127).astype(np.float32)
    return mask[..., np.newaxis]

def load_data(a_dir, b_dir, label_dir):
    X1, X2, Y = [], [], []
    a_paths = sorted(glob(os.path.join(a_dir, '*.png')))
    for path in a_paths:
        filename = os.path.basename(path)
        X1.append(load_image(os.path.join(a_dir, filename)))
        X2.append(load_image(os.path.join(b_dir, filename)))
        Y.append(load_mask(os.path.join(label_dir, filename)))
    return np.array(X1), np.array(X2), np.array(Y)
