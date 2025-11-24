import os
import sys
import unittest
import tempfile
import numpy as np
import skimage.io as skio

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.image_tools import pad_image
from main.patchmatch import PatchMatch


def _pad_spatial_shape(shape_hw_or_hwc, r):
    """Return padded spatial H,W given an image shape and pad radius r."""
    H, W = shape_hw_or_hwc[:2]
    return (H + 2 * r, W + 2 * r)

def _shift_image_reflect(A, dy, dx):
    H, W = A.shape[:2]
    r = max(abs(dy), abs(dx))
    Ap = pad_image(A, r)
    sy = r - dy           
    sx = r - dx
    return Ap[sy:sy+H, sx:sx+W]

def _fraction_within_tol(ofs, tgt_dy, tgt_dx, tol=0):
    """ofs: (H,W,2) with dy in [:,:,0] and dx in [:,:,1]."""
    di = ofs[..., 0]
    dj = ofs[..., 1]
    return np.mean((np.abs(di - tgt_dy) <= tol) & (np.abs(dj - tgt_dx) <= tol))

class TestGroundTruthPatchMatchClass(unittest.TestCase):

    def test_identity_gray_offsets_near_zero(self):
        with tempfile.TemporaryDirectory() as td:
            np.random.seed(0)
            A = (np.random.rand(64, 64) * 255).astype(np.uint8)
            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, A.copy())

            pm = PatchMatch(fA, fB, patch_size=7, iterations=5, seed=0)
            ofs = pm.run()

            frac_exact = _fraction_within_tol(ofs, 0, 0, tol=0)
            self.assertGreater(frac_exact, 0.90)
            frac_loose = _fraction_within_tol(ofs, 0, 0, tol=1)
            self.assertGreater(frac_loose, 0.90)

    def test_identity_color_offsets_near_zero(self):
        with tempfile.TemporaryDirectory() as td:
            np.random.seed(1)
            A = (np.random.rand(48, 50, 3) * 255).astype(np.uint8)
            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, A.copy())

            pm = PatchMatch(fA, fB, patch_size=7, iterations=5, seed=1)
            ofs = pm.run()

            frac_exact = _fraction_within_tol(ofs, 0, 0, tol=0)
            self.assertGreater(frac_exact, 0.90)
            frac_loose = _fraction_within_tol(ofs, 0, 0, tol=1)
            self.assertGreater(frac_loose, 0.90)

    def test_constant_translation_gray(self):
        with tempfile.TemporaryDirectory() as td:
            np.random.seed(2)
            A = (np.random.rand(64, 64) * 255).astype(np.uint8)
            dy, dx = 3, -2
            B = _shift_image_reflect(A, dy, dx)

            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, B)

            pm = PatchMatch(fA, fB, patch_size=7, iterations=6, seed=2)
            ofs = pm.run()

            frac_exact = _fraction_within_tol(ofs, dy, dx, tol=0)
            self.assertGreater(frac_exact, 0.90)
            frac_loose = _fraction_within_tol(ofs, dy, dx, tol=1)
            self.assertGreater(frac_loose, 0.90)

    def test_identity_with_noise_gray(self):
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(4)
            A = (rng.random((64, 64)) * 255).astype(np.float32)
            noise = rng.normal(0, 2.0, size=A.shape).astype(np.float32)
            B = np.clip(A + noise, 0, 255).astype(np.uint8)
            A = A.astype(np.uint8)

            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, B)

            pm = PatchMatch(fA, fB, patch_size=7, iterations=6, seed=4)
            ofs = pm.run()

            frac_loose = _fraction_within_tol(ofs, 0, 0, tol=1)
            self.assertGreater(frac_loose, 0.90)

    def test_offsets_within_valid_bounds(self):
        """All matched top-lefts (i+di, j+dj) must be valid in B_padded."""
        with tempfile.TemporaryDirectory() as td:
            A = (np.random.rand(32, 33, 3) * 255).astype(np.uint8)
            B = (np.random.rand(32, 33, 3) * 255).astype(np.uint8)
            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, B)

            p = 9
            r = p // 2
            H2p, W2p = _pad_spatial_shape(B.shape, r)  # padded spatial dims

            pm = PatchMatch(fA, fB, patch_size=p, iterations=2, seed=5)
            ofs = pm.run()

            H, W = A.shape[:2]
            for i in range(H):
                for j in range(W):
                    di, dj = ofs[i, j]
                    mi, mj = i + int(di), j + int(dj)
                    self.assertGreaterEqual(mi, 0)
                    self.assertGreaterEqual(mj, 0)
                    self.assertLessEqual(mi, H2p - p)
                    self.assertLessEqual(mj, W2p - p)


if __name__ == "__main__":
    unittest.main()
