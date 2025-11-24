# copy-paste-detection

A tool to detect operations of copy-pasting regions of an image onto itself. At its core, we use the PatchMatch algorithm.

## Logic of PatchMatch

1. **Initialization**: Random assignment of matches.

2. **Propagation**: We get the offset for the pixels at the top and at the left translated one pixel to the right or to the bottom if the error is less than the current one at the pixel we are examinating. We then propagate backwards (bottom and right).

3. **Random search**: We get the current offset and search randomly in a ever-decreasing radius to find out if we can improve even more our match.

## Detection of Copy–Paste Regions

After PatchMatch converges, each pixel (i, j) has an associated **offset vector** (Δi, Δj) pointing to its best matching patch elsewhere in the image. The detection stage uses this offset field to turn “good matches” into actual copy–paste predictions:

1. **Patch similarity check**  
   For every pixel, we compare its patch with the patch at its matched location using a squared-difference score. Pixels whose score is below `diff_threshold` are considered **copy–paste candidates**.

2. **Reject trivial / local matches**  
   Matches that are too close to the original pixel are usually just natural self-similarity (texture/background). We discard candidates whose offset magnitude is smaller than `min_norm`.

3. **Clean up noise**  
   The binary candidate mask is refined with morphological opening/closing (and optional median filtering on offsets) to remove isolated false positives and fill small gaps in true regions.

4. **Pair regions using offset clustering**  
   Candidate pixels are grouped by **similar offsets** (optionally with spatial coordinates) using DBSCAN. Each cluster represents a coherent copy–paste operation. For every cluster, we color both:
   - the **source region** (where the cluster pixels are), and
   - the **destination region** (cluster pixels shifted by the cluster’s median offset).

## How to Run

Below is a minimal example showing how to run the copy–paste detector on a single image and visualize paired source/copy regions.

1. **Load the image**
2. **Set PatchMatch parameters**
3. **Instantiate the detector**
4. **Set pairing / clustering parameters**
5. **Run detection**

```python
import os
import sys
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import skimage.io
from main.detector import CopyPasteDetector as CPD

# 1) Load image
img_path = "../images/fleur_copy_paste_normal.png"
img = skimage.io.imread(img_path)

# 2) PatchMatch parameters
patch_size = 7        # patch width/height 
iterations = 5        # PatchMatch number of iterations
min_norm = 15         # minimum offset magnitude to ignore trivial matches
diff_threshold = 500  # max patch difference to accept a match as copy–paste

# 3) Create detector and run PatchMatch internally
detector = CPD(img, patch_size, iterations, min_norm, diff_threshold)
img = detector.img    # preprocessed image used by the detector

# 4) Pairing / clustering parameters
cluster_eps = 2.0       # DBSCAN radius in feature space
flat_threshold = 0.05   # discard very flat/background patches
min_cluster_size = 300  # minimum patches per detected region

# 5) Detect paired regions
paired_result = detector.detect_paired_regions(
    cluster_eps=cluster_eps,
    flat_threshold=flat_threshold,
    min_cluster_size=min_cluster_size
)
