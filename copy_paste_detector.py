#!/usr/bin/env python3

from skimage import io as skio
from skimage.color import gray2rgb
from patchmatch import PatchMatch
from utils.image_tools import get_patch, diff
import matplotlib.pyplot as plt
import numpy as np

img_1_path = 'images/suburb_copy_paste_normal.png'
patch_size = 7
iterations = 5
min_norm = 15

DIFF_THRESHOLD = 500.0

print("initiating patch match algorithm")

patch_match = PatchMatch(img_1_path, img_1_path,
                         patch_size, iterations, seed=None, min_norm=min_norm,
                         verbose=True)
offsets = patch_match.run()

print("patch match complete. Identifying copy-paste operations...")

img_1_padded = patch_match.img_1_padded
h, w = patch_match.img_1.shape[:2]
diffs = np.zeros((h, w))

for i in range(h):
    for j in range(w):
        patch_1 = get_patch(img_1_padded, i, j, patch_size)
        patch_2 = get_patch(img_1_padded, i + offsets[i, j][0], j + offsets[i, j][1], patch_size)
        diffs[i, j] = diff(patch_1, patch_2)

mask = diffs < DIFF_THRESHOLD

output_image = patch_match.img_1.copy()
if output_image.ndim == 2:
    output_image = gray2rgb(output_image)
elif output_image.ndim == 3 and output_image.shape[2] == 2:
    output_image = gray2rgb(output_image[..., 0])
elif output_image.ndim == 3 and output_image.shape[2] == 4:
    output_image = output_image[..., :3]
output_image = output_image.astype(np.float32)

highlight_color = np.array([255, 0, 0], dtype=np.float32)

output_image[mask] = 0.3 * output_image[mask] + 0.7 * highlight_color

output_image = np.clip(output_image, 0, 255)

plt.figure(figsize=(12, 6))

img_display = patch_match.img_1
if img_display.ndim == 3:
    if img_display.shape[2] == 2:
        img_display = img_display[..., 0]
    elif img_display.shape[2] == 4:
        img_display = img_display[..., :3]
if img_display.ndim == 2:
    plt.subplot(1, 2, 1)
    plt.imshow(img_display, cmap='gray')
else:
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(img_display / np.max(img_display), 0, 1))
plt.title("Original Image 1")
plt.axis('off')

plt.subplot(1, 2, 2)
output_display = output_image
if output_display.ndim == 3 and output_display.shape[2] == 2:
    output_display = output_display[..., 0]
if output_display.ndim == 3 and output_display.shape[2] == 4:
    output_display = output_display[..., :3]
plt.imshow(output_image.astype(np.uint8))
plt.title(f"Detected Regions (Threshold = {DIFF_THRESHOLD})")
plt.axis('off')

plt.tight_layout()
plt.show()

output_filename = "copy_paste_detections.png"
output_image_save = np.clip(output_image, 0, 255).astype(np.uint8)
skio.imsave(output_filename, output_image_save)
print(f"Saved result to {output_filename}")
