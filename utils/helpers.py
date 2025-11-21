import os
import platform
import tempfile
from skimage import io as skio
import matplotlib.pyplot as plt
import numpy as np

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

def detection_results(original_image_path, source_mask, copy_mask):

    img = skio.imread(original_image_path)

    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3] 
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img *= 255.0

    source_highlight_color = np.array([0, 255, 0], dtype=np.float32)
    copy_highlight_color = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.45 

    out = img.copy()
    source_bool = source_mask.astype(bool)
    if source_bool.any():
        out[source_bool] = (1-alpha) * out[source_bool] + alpha * source_highlight_color
    copy_bool = copy_mask.astype(bool)
    if copy_bool.any():
        out[copy_bool] = (1-alpha) * out[copy_bool] + alpha * copy_highlight_color

    return out.astype(np.uint8)

def display_results(original_image_path, detection_image, diff_threshold):
    plt.figure(figsize=(12, 6))

    img_display = skio.imread(original_image_path)

    if img_display.ndim == 2:
        img_display = gray2rgb(img_display)

    elif img_display.ndim == 3:
        if img_display.shape[2] == 2:
            img_display = gray2rgb(img_display[..., 0])
        elif img_display.shape[2] == 4:
            img_display = img_display[..., :3]

    img_display = img_display.astype(np.float32)
    if img_display.max() <= 1.0:
        img_display *= 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(img_display / np.max(img_display), 0, 1))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    output_display = detection_image
    if output_display.ndim == 3 and output_display.shape[2] == 2:
        plt.imshow(img_display, cmap='gray')
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(img_display / np.max(img_display), 0, 1))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    output_display = detection_image
    if output_display.ndim == 3 and output_display.shape[2] == 2:
        output_display = output_display[..., 0]
    elif output_display.ndim == 3 and output_display.shape[2] == 4:
        output_display = output_display[..., :3]

    plt.imshow(output_display.astype(np.uint8))
    plt.title(f"Detected Regions (Threshold = {diff_threshold})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def gray2rgb(gray_image):
    return np.stack((gray_image,)*3, axis=-1)
