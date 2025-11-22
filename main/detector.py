#!/usr/bin/env python3

from scipy.ndimage import label, binary_dilation, binary_opening, find_objects, median_filter, binary_closing
from sklearn.cluster import DBSCAN
from skimage.color import rgb2gray, hsv2rgb
from main.patchmatch import PatchMatch
from utils.image_tools import get_patch, diff
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

class CopyPasteDetector:
    def __init__(self, img, patch_size, iterations, min_norm, diff_threshold):
        self.patch_size = patch_size
        self.iterations = iterations
        # self.min_norm = max(min_norm, patch_size * 2) # avoid local patching
        self.min_norm = min_norm
        self.diff_threshold = diff_threshold
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

    def detect_paired_regions(
            self,
            cluster_eps=2.0,
            flat_threshold=0.0,
            clustering=True,
            min_cluster_size=50,
            median_filtering=True
    ):
        """
        Identify source region and its copies by clustering offset vectors

        returns: output_mask: (H, W, 3) RGB image with colored pairs
        """
        h, w = self.offsets.shape[:2]

        if median_filtering:
            # in noisy images, since the offsets can have slight differences,
            # we uniformize then by applying a median filter
            filt_col = median_filter(self.offsets[..., 0], size=self.patch_size)
            filt_lin = median_filter(self.offsets[..., 1], size=self.patch_size)
            self.offsets = np.stack([filt_col, filt_lin], axis=-1)

        h_pad, w_pad = self.patchmatch.img_1_padded.shape[:2]
        grid = np.moveaxis(np.indices((h, w)), 0, -1)
        max_valid = np.array([h_pad, w_pad]) - self.patch_size
        self.offsets = np.clip(self.offsets, -grid, max_valid - grid)

        # threshold to avoid detecting offsets in flat regions (background)
        if self.img.max() > 1.0:
            flat_threshold *= 255
        offset_magnitude = np.sqrt(self.offsets[..., 0]**2 + self.offsets[..., 1]**2)
        diffs = self.compute_differences()
        copy_paste_candidates = (diffs < self.diff_threshold) & (offset_magnitude >= self.min_norm)
        copy_paste_candidates = binary_opening(copy_paste_candidates, structure=np.ones((5, 5)))

        valid_indices = []
        valid_offsets = []
        lin_coords, col_coords = np.where(copy_paste_candidates)

        # use standard deviation to discard regions that are too flat
        for lin, col in zip(lin_coords, col_coords):
            patch = self.img[lin:lin + self.patch_size, col:col + self.patch_size]
            if np.std(patch) >= flat_threshold:
                valid_indices.append((lin, col))
                valid_offsets.append(self.offsets[lin, col])

        if not valid_offsets:
            return np.zeros((h, w, 3), dtype=np.float32)
        valid_offsets = np.array(valid_offsets)
        valid_indices = np.array(valid_indices)

        output_visualization = np.zeros((h, w, 3), dtype=np.float32)

        if clustering:
            # cluster offsets that are acceptably similar (differ at most [eps] from one another),
            # as in a noisy image we may have some variance between offsets regarding the same copied-pasted region
            clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_size).fit(valid_offsets)
            labels = clustering.labels_

            # -1 means "noise"
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            if len(unique_labels) == 0:
                return np.zeros((h, w, 3), dtype=np.float32)

            output_visualization = np.zeros((h, w, 3), dtype=np.float32)

            colors = hsv2rgb(np.array(
                [[[i/len(unique_labels), 1.0, 1.0] for i in range(len(unique_labels))]]
            ))[0]

            for idx, label_id in enumerate(unique_labels):
                current_color = colors[idx]

                mask_indices = (labels == label_id)
                cluster_pixels = valid_indices[mask_indices]
                cluster_offsets = valid_offsets[mask_indices]

                object_mask = np.zeros((h, w), dtype=bool)
                for (plin, pcol) in cluster_pixels:
                    object_mask[plin, pcol] = True

                # morphological closing to try to fill the gaps inside a detected region,
                # which may contain discontinuities (specially in a noisy image)
                object_mask = binary_closing(object_mask, structure=np.ones((self.patch_size, self.patch_size)))

                filled_lin, filled_col = np.where(object_mask)

                # set offset of filled pixels as the median of the cluster
                avg_offset = np.median(cluster_offsets, axis=0).astype(int)
                dlin, dcol = avg_offset

                for plin, pcol in zip(filled_lin, filled_col):
                    output_visualization[plin, pcol] = current_color
                    dest_lin, dest_col = int(plin + dlin), int(pcol + dcol)
                    if 0 <= dest_lin < h and 0 <= dest_col < w:
                        output_visualization[dest_lin, dest_col] = current_color
        else:
            # color by angle instead of by ID
            angles = np.arctan2(valid_offsets[..., 1], valid_offsets[..., 0])

            hues = (angles + np.pi) / (2 * np.pi)

            for i, (lin, col) in enumerate(valid_indices):
                rgb_color = hsv2rgb(np.array([[[hues[i], 1.0, 1.0]]]))[0][0]
                output_visualization[lin, col] = rgb_color

                dest_lin, dest_col = lin + int(valid_offsets[i][0]), col + int(valid_offsets[i][1])
                if 0 <= dest_lin < h and 0 <= dest_col < w:
                    output_visualization[dest_lin, dest_col] = rgb_color

        return output_visualization

    def visualize_offsets(self):
        """
        Visualizes the raw offset field using HSV color space.
        Direction -> Hue, Magnitude -> Saturation/Value
        """
        h, w = self.offsets.shape[:2]

        magnitude = np.sqrt(self.offsets[..., 0]**2 + self.offsets[..., 1]**2)
        angle = np.arctan2(self.offsets[..., 1], self.offsets[..., 0])

        hue = (angle + np.pi) / (2 * np.pi)
        saturation = np.ones((h, w))
        value = magnitude / (magnitude.max() + 1e-5)

        hsv_img = np.stack((hue, saturation, value), axis=-1)
        rgb_img = hsv2rgb(hsv_img)

        return (rgb_img * 255).astype(np.uint8)

    def compute_differences(self):
        img_padded = self.patchmatch.img_1_padded
        h, w = self.patchmatch.img_1.shape[:2]
        diffs = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                patch_1 = get_patch(img_padded, i, j, self.patch_size)
                patch_2 = get_patch(img_padded, i + self.offsets[i, j][0], j + self.offsets[i, j][1], self.patch_size)
                diffs[i, j] = diff(patch_1, patch_2, np.inf)
        return diffs
    
    def detect_copy_paste(self, min_norm=10):
        offsets_magnitude = np.sqrt(self.offsets[..., 0]**2 + self.offsets[..., 1]**2)
        diffs = self.compute_differences()
        mask = (diffs < self.diff_threshold) & (offsets_magnitude >= self.min_norm)
        return mask

    def detect_source_and_copy(self):
        """
        Differentiate between original and copy-pasted regions by calculating
        boundary discontinuities and comparing them

        returns: source_mask, copy_mask
        """
        raw_mask = self.detect_copy_paste()
        # opening to remove thin noise bridges between elements
        clean_mask = binary_opening(raw_mask, structure=np.ones((3, 3)))

        labeled_array, num_features = label(clean_mask, structure=np.ones((3, 3)))
        if num_features < 2:
            return np.zeros_like(raw_mask), raw_mask

        region_scores = {}
        img = self.patchmatch.img_1
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[..., :3]
            img_gray = rgb2gray(img)
        else:
            img_gray = img

        valid_regions = []
        for region_id in range(1, num_features + 1):
            blob_mask = (labeled_array == region_id)

            # consider minimum area for copy-paste operation, to
            # avoid detecting areas that are similar by coincidence
            if np.sum(blob_mask) < 50:
                continue

            # dilate in two iterations to give us a reasonably large
            # boundary ring to calculate standard deviation
            dilated = binary_dilation(blob_mask, iterations=2)
            boundary_ring = dilated & ~blob_mask
            boundary_vals = img_gray[boundary_ring]
            if len(boundary_vals) == 0:
                score = 0
            else:
                score = np.std(boundary_vals)

            region_scores[region_id] = score
            valid_regions.append(region_id)

        if not valid_regions:
            return np.zeros_like(raw_mask), raw_mask

        scores = [region_scores[rid] for rid in valid_regions]
        if len(valid_regions) == 2:
            r1, r2 = valid_regions
            if region_scores[r1] > region_scores[r2]:
                copy_ids = [r1]
                source_ids = [r2]
            else:
                copy_ids = [r2]
                source_ids = [r1]
        else:
            # if many regions, take the mean of the score and split between sources / copies
            mean_score = np.mean(scores)
            source_ids = []
            copy_ids = []
            for rid in valid_regions:
                if region_scores[rid] > mean_score:
                    copy_ids.append(rid)
                else:
                    source_ids.append(rid)

        copy_mask = np.isin(labeled_array, copy_ids)
        source_mask = np.isin(labeled_array, source_ids)

        return source_mask, copy_mask
