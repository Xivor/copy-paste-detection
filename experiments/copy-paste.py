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
    img_path = "../images/lena_inpainted_4.png"

    patch_size = 7
    iterations = 5
    min_norm = 5
    diff_threshold = 1500
    img = skio.imread(img_path)
    img_orig = img.copy()

    detector = CPD(img, patch_size, iterations, min_norm, diff_threshold)
    # output_visualization = detector.visualize_offsets()
    # display_results(img, output_visualization)

    paired_result = detector.detect_paired_regions(
        cluster_eps=3.0,
        flat_threshold=0.01,
        min_cluster_size=100
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
    # skio.imsave("fleur_with_flat_threshold.png", overlay)
    display_results(img_orig, overlay)


def detection_noisy_image():
    img_path = "../images/lena_inpainted_4_impulse.png"
    img = skio.imread(img_path)
    patch_size = 7
    iterations = 5
    min_norm = 5
    diff_threshold = 100.0
    img_orig = img.copy()

    detector = CPD(img, patch_size, iterations, min_norm, diff_threshold, median_filtering=True)
    img = detector.img
    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.shape[2] == 4:
        img = img[..., :3]
    cluster_eps=1.0
    flat_threshold=0.0
    min_cluster_size=10
    paired_result = detector.detect_paired_regions(
        cluster_eps,
        flat_threshold,
        clustering=True,
        min_cluster_size=min_cluster_size,
        median_filtering=True,
        median_filter_ksize=3,
        spatial_weight=0.1
    )
    result = detection_results(img, paired_result)
    display_results(img_orig, result)
    # skio.imsave("suburbs_with_clustering.png", result)


# detection_normal_image()
detection_noisy_image()
