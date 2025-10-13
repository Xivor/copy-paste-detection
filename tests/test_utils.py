from utils.image_tools import pad_image, get_patch, diff, prepare_images
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


class TestPadImage(unittest.TestCase):
    def test_grayscale_shape_and_center_unchanged(self):
        im = np.arange(25, dtype=np.float32).reshape(5, 5)  # 5x5 grayscale
        r = 2
        ip = pad_image(im, r)
        # shape grows by 2r in H and W
        self.assertEqual(ip.shape, (5 + 2 * r, 5 + 2 * r))
        # center region equals original
        self.assertTrue(np.allclose(ip[r:r + 5, r:r + 5], im))

    def test_grayscale_reflect_corners(self):a 
        im = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=np.float32)
        r = 1
        ip = pad_image(im, r)  # shape (5,5)
        # reflect padding: top-left equals original [1,1]; bottom-right equals
        # original [-2,-2]
        self.assertEqual(ip[0, 0], im[1, 1])
        self.assertEqual(ip[-1, -1], im[-2, -2])

    def test_color_shape_and_no_channel_pad(self):
        im = np.random.rand(3, 4, 3).astype(np.float32)  # H=3, W=4, C=3
        r = 2
        ip = pad_image(im, r)
        self.assertEqual(ip.shape, (3 + 2 * r, 4 + 2 * r, 3))
        # center region equals original (no channel padding)
        self.assertTrue(np.allclose(ip[r:r + 3, r:r + 4, :], im))

    def test_invalid_dims_raises(self):
        im = np.arange(10, dtype=np.float32)  # 1D
        with self.assertRaises(ValueError):
            _ = pad_image(im, 1)


class TestGetPatch(unittest.TestCase):
    def test_get_patch_middle_region(self):
        im = np.arange(100, dtype=np.float32).reshape(10, 10)
        p = 4
        patch = get_patch(im, 3, 5, p)  # rows 3..6, cols 5..8
        self.assertEqual(patch.shape, (p, p))
        self.assertTrue(np.allclose(patch, im[3:3 + p, 5:5 + p]))

    def test_get_patch_with_padding_alignment(self):
        # Verify that the center of the returned patch corresponds to the
        # original pixel
        im = np.arange(9, dtype=np.float32).reshape(3, 3)
        p = 3
        r = p // 2
        ip = pad_image(im, r)

        # Patch "centered" at original (0,0) comes from top-left (0,0) in
        # padded
        patch00 = get_patch(ip, 0, 0, p)
        self.assertEqual(patch00.shape, (p, p))
        self.assertEqual(patch00[r, r], im[0, 0])

        # Patch centered at original (1,1) comes from top-left (1,1) in padded
        patch11 = get_patch(ip, 1, 1, p)
        self.assertEqual(patch11[r, r], im[1, 1])


class TestDiff(unittest.TestCase):
    def test_diff_zero_for_identical(self):
        a = np.zeros((5, 5), dtype=np.float32)
        self.assertEqual(diff(a, a), 0.0)

    def test_diff_known_value(self):
        a = np.array([[0, 1],
                      [2, 3]], dtype=np.float32)
        b = np.array([[1, 1],
                      [0, 3]], dtype=np.float32)
        # squared diffs: (1-0)^2 + (1-1)^2 + (0-2)^2 + (3-3)^2 = 1 + 0 + 4 + 0
        # = 5
        self.assertEqual(diff(a, b), 5.0)

    def test_diff_handles_int_and_float(self):
        a = (np.random.rand(4, 4) * 255).astype(np.uint8)
        b = (np.random.rand(4, 4) * 255).astype(np.uint8)
        # should not raise and should return a float sum
        d = diff(a, b)
        self.assertIsInstance(d, (float, np.floating))


class TestPrepareImages(unittest.TestCase):
    def test_prepare_images_grayscale(self):
        with tempfile.TemporaryDirectory() as td:
            # Create small grayscale images and save
            A = (np.random.rand(8, 9) * 255).astype(np.uint8)
            B = (np.random.rand(10, 7) * 255).astype(np.uint8)
            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, B)

            p = 5
            img_1, img_2, img_1_p, img_2_p, maxdim = prepare_images(fA, fB, p)

            # dtypes
            self.assertEqual(img_1.dtype, np.float32)
            self.assertEqual(img_2.dtype, np.float32)
            self.assertEqual(img_1_p.dtype, np.float32)
            self.assertEqual(img_2_p.dtype, np.float32)

            # shapes
            self.assertEqual(img_1.shape, A.shape)
            self.assertEqual(img_2.shape, B.shape)
            r = p // 2
            self.assertEqual(
                img_1_p.shape,
                (A.shape[0] + 2 * r,
                 A.shape[1] + 2 * r))
            self.assertEqual(
                img_2_p.shape,
                (B.shape[0] + 2 * r,
                 B.shape[1] + 2 * r))

            # max dim uses spatial dims only
            self.assertEqual(maxdim, max(B.shape[:2]))

            # center region equals originals
            self.assertTrue(np.allclose(
                img_1_p[r:r + A.shape[0], r:r + A.shape[1]], img_1))
            self.assertTrue(np.allclose(
                img_2_p[r:r + B.shape[0], r:r + B.shape[1]], img_2))

    def test_prepare_images_color(self):
        with tempfile.TemporaryDirectory() as td:
            # Create small color images and save
            A = (np.random.rand(6, 7, 3) * 255).astype(np.uint8)
            B = (np.random.rand(5, 9, 3) * 255).astype(np.uint8)
            fA = os.path.join(td, "A.png")
            fB = os.path.join(td, "B.png")
            skio.imsave(fA, A)
            skio.imsave(fB, B)

            p = 7
            img_1, img_2, img_1_p, img_2_p, maxdim = prepare_images(fA, fB, p)

            # dtypes
            self.assertEqual(img_1.dtype, np.float32)
            self.assertEqual(img_2.dtype, np.float32)
            self.assertEqual(img_1_p.dtype, np.float32)
            self.assertEqual(img_2_p.dtype, np.float32)

            # shapes
            self.assertEqual(img_1.shape, A.shape)
            self.assertEqual(img_2.shape, B.shape)
            r = p // 2
            self.assertEqual(
                img_1_p.shape, (A.shape[0] + 2 * r, A.shape[1] + 2 * r, 3))
            self.assertEqual(
                img_2_p.shape, (B.shape[0] + 2 * r, B.shape[1] + 2 * r, 3))

            # max dim uses spatial dims only
            self.assertEqual(maxdim, max(B.shape[:2]))

            # center region equals originals
            self.assertTrue(np.allclose(
                img_1_p[r:r + A.shape[0], r:r + A.shape[1], :], img_1))
            self.assertTrue(np.allclose(
                img_2_p[r:r + B.shape[0], r:r + B.shape[1], :], img_2))


if __name__ == "__main__":
    unittest.main()
