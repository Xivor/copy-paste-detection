import numpy as np
from skimage import io as skio
from utils.image_tools import get_patch, diff, prepare_images
class PatchMatch():
    def __init__(self, img_1_fp, img_2_fp, patch_size=10, iterations=5, seed=None, min_norm=0):
        self.img_1_fp = img_1_fp
        self.img_2_fp = img_2_fp
        self.patch_size = patch_size
        self.iterations = iterations
        self.seed = seed
        self.min_norm = min_norm
        self.rng = np.random.default_rng(seed)
        self.img_1, self.img_2, self.img_1_padded, self.img_2_padded, self.img_2_max_dim = prepare_images(img_1_fp, img_2_fp, patch_size)
    
    def _initialization(self):
        H, W = self.img_1.shape[:2]
        offsets = np.zeros((H, W, 2), dtype=np.int32)
        for i in range(H):
            for j in range(W):
                while True:
                    rand_i = self.rng.integers(0, self.img_2.shape[0])
                    rand_j = self.rng.integers(0, self.img_2.shape[1])
                    if np.linalg.norm(np.array([rand_i - i, rand_j - j])) >= self.min_norm:
                        offsets[i, j] = [rand_i - i, rand_j - j]
                        break
        return offsets

    def _propagation(self, offsets, lin, col, forward=True):
        d = -1 if forward else 1
        patch = get_patch(self.img_1_padded, lin, col, self.patch_size)

        current_offset = offsets[lin, col]
        match_lin, match_col = [lin, col] + current_offset
        best_diff = diff(patch, get_patch(self.img_2_padded, match_lin, match_col, self.patch_size))

        # top / bottom neighbor
        if 0 <= lin + d < offsets.shape[0]:
            neighbor_offset = offsets[lin + d, col]
            if np.linalg.norm(neighbor_offset) >= self.min_norm:
                match_lin_n1, match_col_n1 = [lin, col] + neighbor_offset
                if 0 <= match_lin_n1 < self.img_2_padded.shape[0] - self.patch_size and \
                   0 <= match_col_n1 < self.img_2_padded.shape[1] - self.patch_size:
                    diff_1 = diff(patch, get_patch(self.img_2_padded, match_lin_n1, match_col_n1, self.patch_size))
                    if diff_1 < best_diff:
                        current_offset = neighbor_offset
                        best_diff = diff_1
        # left / right neighbor
        if 0 <= col + d < offsets.shape[1]:
            neighbor_offset = offsets[lin, col + d]
            if np.linalg.norm(neighbor_offset) >= self.min_norm:
                match_lin_n2, match_col_n2 = [lin, col] + neighbor_offset
                if 0 <= match_lin_n2 < self.img_2_padded.shape[0] - self.patch_size and \
                   0 <= match_col_n2 < self.img_2_padded.shape[1] - self.patch_size:
                    diff_2 = diff(patch, get_patch(self.img_2_padded, match_lin_n2, match_col_n2, self.patch_size))
                    if diff_2 < best_diff:
                        current_offset = neighbor_offset
                        best_diff = diff_2

        offsets[lin, col] = current_offset

    def _random_search(self, offsets, lin, col, rad):
        current_offset = offsets[lin, col]
        best_match_lin, best_match_col = [lin, col] + current_offset

        patch = get_patch(self.img_1_padded, lin, col, self.patch_size)
        best_diff = diff(patch, get_patch(self.img_2_padded, best_match_lin, best_match_col, self.patch_size))

        h_pad, w_pad = self.img_2_padded.shape[:2]

        while rad >= 1:
            search_min_lin = max(0, best_match_lin - rad)
            search_max_lin = min(h_pad - self.patch_size, best_match_lin + rad)
            search_min_col = max(0, best_match_col - rad)
            search_max_col = min(w_pad - self.patch_size, best_match_col + rad)

            if search_min_lin > search_max_lin or search_min_col > search_max_col:
                rad //= 2
                continue

            rand_lin = self.rng.integers(search_min_lin, search_max_lin + 1)
            rand_col = self.rng.integers(search_min_col, search_max_col + 1)

            random_diff = diff(patch, get_patch(self.img_2_padded, rand_lin, rand_col, self.patch_size))
            random_offset = np.array([rand_lin - lin, rand_col - col])

            if random_diff < best_diff and np.linalg.norm(random_offset) >= self.min_norm:
                best_diff = random_diff
                offsets[lin, col] = random_offset

            rad //= 2
        
    def run(self):
        offsets = self._initialization()

        for iteration in range(self.iterations):
            for i in range(self.img_1.shape[0]):  # lines
                for j in range(self.img_1.shape[1]):  # columns
                    self._propagation(offsets, i, j, True)
                    self._random_search(offsets, i, j, self.img_2_max_dim)
            for i in range(self.img_1.shape[0] - 1, -1, -1):
                for j in range(self.img_1.shape[1] - 1, -1, -1):
                    self._propagation(offsets, i, j, False)
                    self._random_search(offsets, i, j, self.img_2_max_dim)
        return offsets
