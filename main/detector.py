#!/usr/bin/env python3

from scipy.ndimage import label, binary_dilation, binary_opening, find_objects, median_filter, binary_closing
from sklearn.cluster import DBSCAN
from skimage.color import rgb2gray, hsv2rgb
from main.patchmatch import PatchMatch
from utils.image_tools import get_patch, diff
import matplotlib.pyplot as plt
import numpy as np

class CopyPasteDetector:
    def __init__(self, img, patch_size, iterations, min_norm, diff_threshold):
        self.patch_size = patch_size
        self.iterations = iterations
        self.min_norm = min_norm
        self.diff_threshold = diff_threshold
        self.patchmatch = PatchMatch(img_path, img_path,
                                     patch_size, iterations, seed=None, min_norm=min_norm,
        self.patchmatch = PatchMatch(img, img,
                                     patch_size, iterations, seed=None, min_norm=self.min_norm,
                                     verbose=True)
        self.offsets = self.patchmatch.run()
        self.img = self.patchmatch.img_1
        if self.img.ndim == 3:
            if self.img.shape[2] == 4:
                self.img = self.img[..., :3]
            elif self.img.shape[2] == 2:
                self.img = self.img[..., 0]

    def compute_differences(self):
        img_padded = self.patchmatch.img_1_padded
        h, w = self.patchmatch.img_1.shape[:2]
        diffs = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                patch_1 = get_patch(img_padded, i, j, self.patch_size)
                patch_2 = get_patch(img_padded, i + self.offsets[i, j][0], j + self.offsets[i, j][1], self.patch_size)
                diffs[i, j] = diff(patch_1, patch_2)
        return diffs
    
    def detect_copy_paste(self):
        diffs = self.compute_differences()
        mask = diffs < self.diff_threshold
        return mask
