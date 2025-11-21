#!/usr/bin/env python3
import os
import sys
import numpy as np
from skimage import io as skio
from skimage.color import gray2rgb
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import gaussian_filter, median_filter

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.patchmatch import PatchMatch
from main.detector import CopyPasteDetector as CPD
from utils.helpers import detection_results, display_results

def detection_normal_image():
    img_path = "../images/fleur_copy_paste_normal.png"

    patch_size = 7
    iterations = 5
    min_norm = 15
    diff_threshold = 500
    img = skio.imread(img_path)

    detector = CPD(img, patch_size, iterations, min_norm, diff_threshold)
    output_visualization = detector.visualize_offsets()
    display_results(img_path, output_visualization, diff_threshold)

    paired_result = detector.detect_paired_regions(
        cluster_eps=2.0,
        flat_threshold=0.05,
        min_cluster_size=300
    )
    paired_result = (paired_result * 255).astype(np.uint8)

    if img.ndim == 2:
        img = gray2rgb(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        elif img.shape[2] == 2:
            img = gray2rgb(img[..., 0])

    mask = np.any(paired_result > 0, axis = -1)
    overlay = img.copy()
    overlay[mask] = paired_result[mask]
    # source_mask, copy_mask = detector.detect_source_and_copy()
    # result = detection_results(img_path, source_mask, copy_mask)
    display_results(img_path, overlay, diff_threshold)


def detection_noisy_image():
    img_path = "../images/fleur_copy_paste_gauss.png"

    patch_size = 7
    iterations = 5
    min_norm = 15
    diff_threshold = 300
    img = skio.imread(img_path)

    # Gaussian filter
    img = gaussian_filter(img, sigma=1.5)

    # NLM filter
    # sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    # img = denoise_nl_means(
    #     img,
    #     h=.1,
    #     fast_mode=True,
    #     patch_size=5,
    #     patch_distance=11,
    #     channel_axis=-1
    # )
    # if img.dtype != np.uint8:
    #     img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Median filter
    #if img.ndim == 3:
    #    if img.shape[2] == 4:
    #        img = img[..., :3]
    #    elif img.shape[2] == 2:
    #        img = img[..., 0]
    #if img.ndim == 3:
    #    for i in range(img.shape[2]):
    #        img[:, :, i] = median_filter(img[:, :, i], size=3)
    #else:
    #    img = median_filter(img, size=5)

    detector = CPD(img, patch_size, iterations, min_norm, diff_threshold)
    output_visualization = detector.visualize_offsets()
    display_results(img_path, output_visualization, diff_threshold)

    paired_result = detector.detect_paired_regions(
        cluster_eps=2.0,
        flat_threshold=0.1,
        min_cluster_size=300
    )
    paired_result = (paired_result * 255).astype(np.uint8)

    if img.ndim == 2:
        img = gray2rgb(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        elif img.shape[2] == 2:
            img = gray2rgb(img[..., 0])
    mask = np.any(paired_result > 0, axis = -1)
    overlay = img.copy()
    overlay[mask] = paired_result[mask]
    # source_mask, copy_mask = detector.detect_source_and_copy()
    # result = detection_results(img_path, source_mask, copy_mask)
    display_results(img_path, overlay, diff_threshold)


detection_noisy_image()
