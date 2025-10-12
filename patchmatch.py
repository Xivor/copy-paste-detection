#!/usr/bin/env python3

import numpy as np
import platform
import tempfile
import os
from skimage import io as skio

# LOGIC OF PATCHMATCH:
# 1. random assignment of matches (initialization)
# 2. propagation: we get the offset for the pixels at the top and at the left translated
#    one pixel to the right or to the bottom if the error is less than the current one at
#    the pixel we are examinating. We then propagate backwards (bottom and right)
# 3. random search: we get the current offset and search randomly in a ever-decreasing
#    radius to find out if we can improve even more our match


###################################################
# HELPER FUNCTIONS
###################################################

def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee
        entre 0 et 255 avant d'Ãªtre sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat

    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase=' '
    elif platform.system()=='Windows':
        #ou windows ; probleme : il faut fermer gimp pour reprendre la main;
        #si vous savez comment faire (commande start ?) je suis preneur
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais pas comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase= ' &'

    if normalise:
        m=im.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=255*imt/M

    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
        imt *= 255

    nomfichier=tempfile.mktemp('TPIMA.png')
    commande=prephrase +nomfichier+endphrase
    imt = imt.astype(np.uint8)
    skio.imsave(nomfichier,imt)
    print(commande)
    os.system(commande)


def get_patch(img, lin, col, patch_size):
    # the patch's center in the original image is the top
    # left corner in the padded image
    lin_end = lin + patch_size
    col_end = col + patch_size
    return img[lin:lin_end, col:col_end]


def diff(patch_1, patch_2):
    diff = np.square(patch_1.astype(np.float32) - patch_2.astype(np.float32))
    return np.sum(diff)


###################################################
# PATCHMATCH
###################################################


def initialization(img_1, img_2, patch_size):
    offsets = np.zeros((img_1.shape[0], img_1.shape[1], 2))
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            rand_i = np.random.randint(0, img_2.shape[0])
            rand_j = np.random.randint(0, img_2.shape[1])
            offsets[i, j] = [rand_i - i, rand_j - j]
    # print(offsets.shape)
    # print(offsets)
    return offsets


def propagation(img_1_padded, img_2_padded, offsets, lin, col, patch_size, forward=True):
    d = -1 if forward else 1
    patch = get_patch(img_1_padded, lin, col, patch_size)

    current_offset = offsets[lin, col]
    match_lin, match_col = [lin, col] + current_offset
    best_diff = diff(patch, get_patch(img_2_padded, match_lin, match_col, patch_size))

    # left / right neighbor
    if 0 <= lin + d < offsets.shape[0]:
        neighbor_offset = offsets[lin + d, col]
        match_lin_n1, match_col_n1 = [lin, col] + neighbor_offset
        if 0 <= match_lin_n1 < img_2_padded.shape[0] - patch_size and \
           0 <= match_col_n1 < img_2_padded.shape[1] - patch_size:
            diff_1 = diff(patch, get_patch(img_2_padded, match_lin_n1, match_col_n1, patch_size))
            if diff_1 < best_diff:
                current_offset = neighbor_offset
                best_diff = diff_1

    # top / bottom  neighbor
    if 0 <= col + d < offsets.shape[1]:
        neighbor_offset = offsets[lin, col + d]
        match_lin_n2, match_col_n2 = [lin, col] + neighbor_offset
        if 0 <= match_lin_n2 < img_2_padded.shape[0] - patch_size and \
           0 <= match_col_n2 < img_2_padded.shape[1] - patch_size:
            diff_2 = diff(patch, get_patch(img_2_padded, match_lin_n2, match_col_n2, patch_size))
            if diff_2 < best_diff:
                current_offset = neighbor_offset

    offsets[lin, col] = current_offset


def random_search(img_1_padded, img_2_padded, offsets, lin, col, patch_size, rad):
    current_offset = offsets[lin, col]
    best_match_lin, best_match_col = [lin, col] + current_offset

    patch = get_patch(img_1_padded, lin, col, patch_size)
    best_diff = diff(patch, get_patch(img_2_padded, best_match_lin, best_match_col, patch_size))

    while rad >= 1:
        search_min_lin = best_match_lin - rad
        search_max_lin = best_match_lin + rad
        search_min_col = best_match_col - rad
        search_max_col = best_match_col + rad

        rand_lin = np.random.randint(search_min_lin, search_max_lin + 1)
        rand_col = np.random.randint(search_min_col, search_max_col + 1)
        rand_lin = np.clip(rand_lin, 0, img_2_padded.shape[0] - patch_size)
        rand_col = np.clip(rand_col, 0, img_2_padded.shape[1] - patch_size)

        random_diff = diff(patch, get_patch(img_2_padded, rand_lin, rand_col, patch_size))

        if random_diff < best_diff:
            best_diff = random_diff
            offsets[lin, col] = [rand_lin - lin, rand_col - col]

        rad //= 2


def patch_match(img_1_fp, img_2_fp, patch_size=10, iterations=5):
    img_1 = skio.imread(img_1_fp)
    img_2 = skio.imread(img_2_fp)
    if img_1.dtype == np.uint8:
        img_1 = img_1.astype(np.float32)
        img_2 = img_2.astype(np.float32)
    img_2_max_dim = max(img_2.shape)
    pad_radius = patch_size // 2
    img_1_padded = np.pad(img_1, pad_radius, mode='reflect')
    img_2_padded = np.pad(img_2, pad_radius, mode='reflect')

    # TEST just to see if everything is right
    # print(img_1.shape)
    # print(img_2.shape)
    # print(offsets)
    # print(offsets.shape)

    # INITIALIZATION

    offsets = initialization(img_1, img_2, patch_size)
    for iteration in range(iterations):
        for i in range(img_1.shape[0]):  # lines
            for j in range(img_1.shape[1]):  # columns
                propagation(img_1_padded, img_2_padded, offsets, i, j, patch_size, True)
                random_search(img_1_padded, img_2_padded, offsets, i, j, patch_size, img_2_max_dim)
        for i in range(img_1.shape[0] - 1, -1, -1):
            for j in range(img_1.shape[1] - 1, -1, -1):
                propagation(img_1_padded, img_2_padded, offsets, i, j, patch_size, False)
                random_search(img_1_padded, img_2_padded, offsets, i, j, patch_size, img_2_max_dim)
    return offsets


offsets = patch_match("images/aerien1.tif", "images/aerien2.tif")
