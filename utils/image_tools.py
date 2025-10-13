import numpy as np
import skimage.io as skio

def pad_image(im, pad_radius):
    # grayscale
    if im.ndim == 2:  
        return np.pad(im, ((pad_radius, pad_radius), (pad_radius, pad_radius)), mode='reflect')
    # color
    elif im.ndim == 3:  
        return np.pad(im, ((pad_radius, pad_radius), (pad_radius, pad_radius), (0, 0)), mode='reflect')
    else:
        raise ValueError("Unsupported image dimensions")


def get_patch(img, lin, col, patch_size):
    # the patch's center in the original image is the top
    # left corner in the padded image
    lin_end = lin + patch_size
    col_end = col + patch_size
    return img[lin:lin_end, col:col_end]


def diff(patch_1, patch_2):
    diff = np.square(patch_1.astype(np.float32) - patch_2.astype(np.float32))
    return np.sum(diff)

def prepare_images(img_1_fp, img_2_fp, patch_size):
    img_1 = skio.imread(img_1_fp).astype(np.float32)
    img_2 = skio.imread(img_2_fp).astype(np.float32)
    H2, W2 = img_2.shape[:2]
    img_2_max_dim = max(H2, W2)
    pad_radius = patch_size // 2
    img_1_padded = pad_image(img_1, pad_radius)
    img_2_padded = pad_image(img_2, pad_radius)
    return img_1, img_2, img_1_padded, img_2_padded, img_2_max_dim