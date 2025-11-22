import numpy as np
import skimage.io as skio
from numba import jit

def pad_image(im, pad_radius):
    # grayscale
    if im.ndim == 2:  
        return np.pad(im, ((pad_radius, pad_radius), (pad_radius, pad_radius)), mode='reflect')
    # color
    elif im.ndim == 3:  
        return np.pad(im, ((pad_radius, pad_radius), (pad_radius, pad_radius), (0, 0)), mode='reflect')
    else:
        raise ValueError("Unsupported image dimensions")


@jit(nopython=True, cache=True)
def get_patch(img, lin, col, patch_size):
    # the patch's center in the original image is the top
    # left corner in the padded image
    lin_end = lin + patch_size
    col_end = col + patch_size
    return img[lin:lin_end, col:col_end]


@jit(nopython=True, cache=True)
def diff(patch_1, patch_2, limit):
    p1 = patch_1.ravel()
    p2 = patch_2.ravel()

    sum = 0.0

    for i in range(p1.shape[0]):
        d = p1[i] - p2[i]
        sum += d * d

        # saves a bit of computational time by discarding
        # this match if part of its quadratic difference is
        # already greater than the best we currently have
        if sum >= limit:
            return limit

    return sum

def prepare_images(img_1, img_2, patch_size):
    H2, W2 = img_2.shape[:2]
    img_2_max_dim = max(H2, W2)
    pad_radius = patch_size // 2
    img_1_padded = pad_image(img_1, pad_radius)
    img_2_padded = pad_image(img_2, pad_radius)
    return img_1_padded, img_2_padded, img_2_max_dim
