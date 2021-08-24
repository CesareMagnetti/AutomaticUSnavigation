import numpy as np
from PIL import Image, ImageOps
import argparse
import os
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='train/test scripts to launch navigation experiments.')
parser.add_argument('--root', '-r', type=str, help='root with images.')
args = parser.parse_args()

if __name__=="__main__":
    image_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if ".png" in f]
    images = np.stack([np.array(ImageOps.grayscale(Image.open(file))) for file in image_files])
    mean_image = images.mean(0)
    std_image = images.std(0)

    image = np.zeros((256,256,3))
    image[..., 0] = std_image
    image[..., 2] = mean_image
    image/=image.max()
    plt.imsave(os.path.join(args.root, "mean_std_image.png"), image)

