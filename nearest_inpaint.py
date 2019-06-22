#! /usr/bin/python

import os, sys
from PIL import Image
from collections import Counter
from random import shuffle


def is_valid(c, max):
    "Ensure that pixel coordinate is not out of range of the image"
    if (c >= 0 and c < max):
        return True
    else:
        return False


def neighbor_pixels(x, y, img,height,width):
    "Find all valid neighboring pixels"
    neighbors = []
    if (is_valid(y + 1, height) and img[x, y + 1][3] == 0):
        neighbors.append([x, y + 1])
    if (is_valid(y - 1, height) and img[x, y - 1][3] == 0):
        neighbors.append([x, y - 1])
    if (is_valid(x + 1, width) and img[x + 1, y][3] == 0):
        neighbors.append([x + 1, y])
    if (is_valid(x - 1, width) and img[x - 1, y][3] == 0):
        neighbors.append([x - 1, y])
    return neighbors


def extract_alpha(img,height,width):
    "Create a list of pixels [[x,y],...] for a given image where pixels are not null"
    alpha = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            if (img[x, y][3] != 0):
                alpha.append([x, y])
            x = x + 1
        y = y + 1
    shuffle(alpha)
    return alpha


def average_rgb(pixels, img):
    "For a given list of pixels [[x,y],...], return the average color as RGB tuple"
    r, g, b = 0, 0, 0
    for p in pixels:
        c = img[p[0], p[1]]
        r += c[0]
        g += c[1]
        b += c[2]
    length = len(pixels)
    if length > 0:
        return (r / length, g / length, b / length)
    else:
        return (0, 0, 0)


def inpainting(img_path, mask_path, save_path):
    image = Image.open(img_path)
    image = image.convert("RGB")
    im = image.load()
    mask = Image.open(mask_path)
    ma = mask.load()

    if (image.size != mask.size):
        print
        "ERROR: Input image and mask have different dimensions!"
        sys.exit(-1)

    width, height = image.size

    tmask = mask.copy()
    tma = tmask.load()

    alpha = extract_alpha(ma,height,width)

    if len(alpha) != 0:
        while len(alpha) > 0:
            for p in alpha:
                neighbors = neighbor_pixels(p[0], p[1], ma,height,width)
                if len(neighbors) > 0:
                    im[p[0], p[1]] = average_rgb(neighbors, im)
                    tma[p[0], p[1]] = 0
            ma = tma
            alpha = extract_alpha(ma)

    image.save(save_path)

if __name__=="__main__":
    img="E:\PyProjects\Sketch-Gated-Inpainting\examples\celeba\images\celeba_01.png"
    mask="E:\PyProjects\Sketch-Gated-Inpainting\examples\celeba\masks\celeba_01.png"
    save="E:\PyProjects\Sketch-Gated-Inpainting\examples\celeba\celeba_01.png"
    inpainting(img, mask, save)