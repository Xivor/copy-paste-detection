#!/usr/bin/env python3

import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image
from skimage import io as skio
import IPython
from skimage.transform import rescale

# LOGIC OF PATCHMATCH:
# 1. random assignment of matches (initialization)
# 2. propagation: we get the offset for the pixels at the top and at the left translated
#    one pixel to the right or to the bottom if the error is less than the current one at
#    the pixel we are examinating.
# 3. random search: we get the current offset and search randomly in a ever-decreasing
#    radius to find out if we can improve even more our match

# make the vector of offsets global? The images as well?

img_1 = None
img_1_padded = None
img_2 = None
img_2_padded = None
offsets = None
patch_size = None

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


def get_patch(img, lin, col):
    # remember: the patch's center in the original image is the top
    # left corner in the padded image
    lin_end = lin + patch_size
    col_end = col + patch_size
    return img[lin:lin_end, col:col_end]


def diff(patch_1, patch_2):
    diff = np.square(patch_1 - patch_2)
    return np.sum(diff)


###################################################
# PATCHMATCH
###################################################


def initialization(img_1, img_2, offsets):
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            a = [i, j]
            b = [np.random.randint(0, img_2.shape[0]), np.random.randint(0, img_2.shape[1])]
            offsets[i, j] = np.array(b) - np.array(a)
    print(offsets.shape)
    print(offsets)


def propagation(img_1, img_2, offsets, lin, col, forward=True):
    if forward:
        d = -1
    else:
        d = 1
    patch_1 = get_patch(img_1, lin, col)
    patch_2 = get_patch(img_2, lin + offsets[lin, col][0], col + offsets[lin, col][1])
    current_diff = diff(patch_1, patch_2)
    # TODO we have to change the verification if it is backwards propagation
    if lin > 0:
        if current_diff > diff(patch_1, get_patch(img_2, lin + offsets[lin+d, col][0], col + offsets[lin+d, col][1])):
            offsets[lin][col] = offsets[lin+d][col]
    if col > 0:
        if current_diff > diff(patch_1, get_patch(img_2, lin + offsets[lin][col+d][0], col + offsets[lin][col+d][1])):
            offsets[lin][col] = offsets[lin][col+d]


def random_search(img_1, img_2, offsets, lin, col, rad):
    if rad < 1:
        return
    random_lin = np.random.randint(lin - rad, lin + rad + 1)
    random_col = np.random.randint(col - rad, col + rad + 1)
    random_lin = np.clip(random_lin, 0, img_2.shape[0] - 1)
    random_col = np.clip(random_col, 0, img_2.shape[1] - 1)
    current_patch = get_patch(img_1, lin, col)
    random_patch = get_patch(
        img_2,
        random_lin,
        random_col
    )
    current_diff = diff(current_patch, get_patch(img_2, lin + offsets[lin][col][0], col + offsets[lin][col][1]))
    random_diff = diff(current_patch, random_patch)
    if current_diff > random_diff:
        offsets[lin][col] = [random_lin - lin, random_col - col]
    return random_search(lin, col, rad//2)


def patch_match(img_1_fp, img_2_fp, patch_size=10, iterations=5):
    img_1 = skio.imread(img_1_fp)
    img_2 = skio.imread(img_2_fp)
    img_2_dim = img_2.shape[0] if img_2.shape[0] > img_2.shape[1] else img_2.shape[1]
    img_1_padded = np.pad(
        img_1,
        ((patch_size//2, patch_size//2),
         (patch_size//2, patch_size//2),
         (0, 0)
         ),
        mode='reflect')
    img_2_padded = np.pad(
        img_2,
        ((patch_size//2, patch_size//2),
         (patch_size//2, patch_size//2),
         (0, 0)
         ),
        mode='reflect')

    # TEST just to see if everything is right
    # print(img_1.shape)
    # print(img_2.shape)
    # print(offsets)
    # print(offsets.shape)

    # INITIALIZATION

    initialization(img_1, img_2, offsets)
    for iteration in range(iterations):
        for i in range(img_1.shape[0]):  # lines
            for j in range(img_1.shape[1]):  # columns
                propagation(img_1_padded, img_2_padded, i, j, True)
                propagation(img_1_padded, img_2_padded, i, j, False)
                random_search(img_1_padded, img_2_padded, i, j, img_2_dim)
    return


patch_match("images/aerien1.tif", "images/aerien2.tif")
