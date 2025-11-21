import numpy as np
from utils.image_tools import get_patch, diff, prepare_images
import time
from tqdm import trange
from numba import jit

class PatchMatch():
    def __init__(self, img_1, img_2, patch_size=10, iterations=5,
                 seed=None, min_norm=0, verbose=False):
        self.img_1 = img_1
        self.img_2 = img_2
        self.patch_size = patch_size
        self.iterations = iterations
        self.seed = seed
        self.min_norm_squared = min_norm**2
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.img_1_padded, self.img_2_padded, self.img_2_max_dim = prepare_images(img_1, img_2, patch_size)
        self.verbose = verbose

    def _initialization(self):
        return _initialization(self.img_1.shape[0], self.img_1.shape[1],
                               self.img_2.shape[0], self.img_2.shape[1], self.min_norm_squared)

    def _propagation(self, offsets, lin, col, forward=True):
        return _propagation(offsets, self.img_1_padded, self.img_2_padded,
                            self.patch_size, self.min_norm_squared, lin, col, forward)

    def _random_search(self, offsets, lin, col, rad, best_diff):
        return _random_search(offsets, self.img_1_padded, self.img_2_padded, self.patch_size,
                              self.min_norm_squared, lin, col, rad, best_diff)

    def run(self):
        if self.verbose:
            start_time = time.perf_counter()
        offsets = self._initialization()

        for iteration in range(self.iterations):
            if self.verbose:
                for i in trange(self.img_1.shape[0], desc=f"Forward pass {iteration+1}..."):  # lines
                    for j in range(self.img_1.shape[1]):  # columns
                        best_diff = self._propagation(offsets, i, j, True)
                        self._random_search(offsets, i, j, self.img_2_max_dim, best_diff)

                for i in trange(self.img_1.shape[0] - 1, -1, -1, desc=f"Reverse pass {iteration+1}..."):
                    for j in range(self.img_1.shape[1] - 1, -1, -1):
                        best_diff = self._propagation(offsets, i, j, False)
                        self._random_search(offsets, i, j, self.img_2_max_dim, best_diff)
            else:
                for i in range(self.img_1.shape[0]):
                    for j in range(self.img_1.shape[1]):
                        best_diff = self._propagation(offsets, i, j, True)
                        self._random_search(offsets, i, j, self.img_2_max_dim, best_diff)

                for i in range(self.img_1.shape[0] - 1, -1, -1):
                    for j in range(self.img_1.shape[1] - 1, -1, -1):
                        best_diff = self._propagation(offsets, i, j, False)
                        self._random_search(offsets, i, j, self.img_2_max_dim, best_diff)

        if self.verbose:
            end_time = time.perf_counter()
            print(f"Total time to run patch match: {end_time - start_time:0.4f} seconds")
        return offsets


@jit(nopython=True, cache=True)
def _initialization(H_img_1, W_img_1, H_img_2, W_img_2, min_norm_squared):
    offsets = np.zeros((H_img_1, W_img_1, 2), dtype=np.int32)
    for i in range(H_img_1):
        for j in range(W_img_1):
            while True:
                rand_i = np.random.randint(0, H_img_2)
                rand_j = np.random.randint(0, W_img_2)
                random_offset = np.array([rand_i - i, rand_j - j])
                if random_offset[0]**2 + random_offset[1]**2 >= min_norm_squared:
                    offsets[i, j] = random_offset
                    break
    return offsets

@jit(nopython=True, cache=True)
def _propagation(offsets, img_1_padded, img_2_padded, patch_size,
                 min_norm_squared, lin, col, forward=True):
    d = -1 if forward else 1
    patch = get_patch(img_1_padded, lin, col, patch_size)

    current_offset = offsets[lin, col]
    match_lin, match_col = np.array([lin, col]) + current_offset
    best_diff = diff(patch, get_patch(img_2_padded, match_lin, match_col, patch_size))

    # top / bottom neighbor
    if 0 <= lin + d < offsets.shape[0]:
        neighbor_offset = offsets[lin + d, col]
        if neighbor_offset[0]**2 + neighbor_offset[1]**2 >= min_norm_squared:
            match_lin_n1, match_col_n1 = np.array([lin, col]) + neighbor_offset
            if 0 <= match_lin_n1 < img_2_padded.shape[0] - patch_size and \
               0 <= match_col_n1 < img_2_padded.shape[1] - patch_size:
                diff_1 = diff(patch, get_patch(img_2_padded, match_lin_n1, match_col_n1, patch_size))
                if diff_1 < best_diff:
                    current_offset = neighbor_offset
                    best_diff = diff_1
    # left / right neighbor
    if 0 <= col + d < offsets.shape[1]:
        neighbor_offset = offsets[lin, col + d]
        if neighbor_offset[0]**2 + neighbor_offset[1]**2 >= min_norm_squared:
            match_lin_n2, match_col_n2 = np.array([lin, col]) + neighbor_offset
            if 0 <= match_lin_n2 < img_2_padded.shape[0] - patch_size and \
               0 <= match_col_n2 < img_2_padded.shape[1] - patch_size:
                diff_2 = diff(patch, get_patch(img_2_padded, match_lin_n2, match_col_n2, patch_size))
                if diff_2 < best_diff:
                    current_offset = neighbor_offset
                    best_diff = diff_2

    offsets[lin, col] = current_offset
    return best_diff

@jit(nopython=True, cache=True)
def _random_search(offsets, img_1_padded, img_2_padded, patch_size,
                   min_norm_squared, lin, col, rad, best_diff):
    current_offset = offsets[lin, col]
    best_match_lin, best_match_col = np.array([lin, col]) + current_offset

    patch = get_patch(img_1_padded, lin, col, patch_size)

    h_pad, w_pad = img_2_padded.shape[:2]

    while rad >= 1:
        search_min_lin = max(0, best_match_lin - rad)
        search_max_lin = min(h_pad - patch_size, best_match_lin + rad)
        search_min_col = max(0, best_match_col - rad)
        search_max_col = min(w_pad - patch_size, best_match_col + rad)

        if search_min_lin > search_max_lin or search_min_col > search_max_col:
            rad //= 2
            continue

        rand_lin = np.random.randint(search_min_lin, search_max_lin + 1)
        rand_col = np.random.randint(search_min_col, search_max_col + 1)

        random_diff = diff(patch, get_patch(img_2_padded, rand_lin, rand_col, patch_size))
        random_offset = np.array([rand_lin - lin, rand_col - col])

        if random_diff < best_diff and random_offset[0]**2 + random_offset[1]**2 >= min_norm_squared:
            best_diff = random_diff
            offsets[lin, col] = random_offset

        rad //= 2
