#!/usr/bin/env python3
import argparse
import numpy as np
from skimage import io as skio
from skimage.util import random_noise

def add_noise(img, noise_type='gaussian', intensity=0.1):
    img = img.astype(np.float32) / 255.0
    if noise_type == 'gaussian':
        noisy = random_noise(img, mode='gaussian', var=intensity**2)
    elif noise_type == 'sp':
        noisy = random_noise(img, mode='s&p', amount=intensity)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'sp'")
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return noisy

def main():
    parser = argparse.ArgumentParser(description="Add noise to an input image.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("noise_type", choices=['gaussian', 'sp'], help="Noise type")
    parser.add_argument("intensity", type=float, help="Noise intensity")
    parser.add_argument("--output", default="noisy_output.png", help="Output file name")
    args = parser.parse_args()

    img = skio.imread(args.image_path)
    noisy = add_noise(img, args.noise_type, args.intensity)
    skio.imsave(args.output, noisy)
    print(f"Saved noisy image to {args.output}")

if __name__ == "__main__":
    main()
