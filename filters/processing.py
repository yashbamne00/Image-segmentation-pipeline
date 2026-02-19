# src/filters.py
import matplotlib.pyplot as plt
import numpy as np

from kernels.sharp import kernel_laplacian_4, kernel_high_boost, kernel_horizontal, kernel_vertical
from kernels.smooth import kernel_mean_3, kernel_gaussian_3, kernel_light_smooth, kernel_strong_smooth
from kernels.edge import kernel_prewitt_x, kernel_prewitt_y, kernel_sobel_x, kernel_sobel_y

def operation(img, type):
    
    h,w,ch= img.shape
    img2 = np.zeros((h, w, ch), dtype=int)
    img = img.astype(int)
    if type == "sharp":
        kernel = kernel_laplacian_4
    elif type == "smooth":
        kernel = kernel_mean_3
    elif type == "edge1":
        kernel = kernel_prewitt_x
    elif type == "edge2":
        kernel = kernel_prewitt_y
    for i in range(1, h-1):
        for j in range(1, w-1):
            for c in range(3):
                val = 0
                val += kernel[0,0] * img[i-1,j-1,c]
                val += kernel[0,1] * img[i-1,j,c]
                val += kernel[0,2] * img[i-1,j+1,c]
                val += kernel[1,0] * img[i,j-1,c]
                val += kernel[1,1] * img[i,j,c]
                val += kernel[1,2] * img[i,j+1,c]
                val += kernel[2,0] * img[i+1,j-1,c]
                val += kernel[2,1] * img[i+1,j,c]
                val += kernel[2,2] * img[i+1,j+1,c]

                img2[i,j,c] = val
    return img2

def threshold(img, t=100):
    h,w,ch = img.shape
    img2 = np.zeros([h,w])
    img = img.astype(int)
    for i in range(0, h):
        for j in range(0, w):
            value = img[i,j]                                      #img[i,j] = intensity not color for color [i,j,c]
            avg = (value[0] + value[1] + value[2] )/3
            if avg > t:
                img2[i,j] = 0
            else:
                img2[i,j] = 1
    return img2

def threshold2(img, t=200):
    h,w,ch = img.shape
    img3 = np.zeros([h,w])
    img = img.astype(int)
    for i in range(0, h):
        for j in range(0, w):
            value = img[i,j]                                      #img[i,j] = intensity not color for color [i,j,c]
            avg = (value[0] + value[1] + value[2] )/3
            if avg > t:
                img3[i,j] = 0
            else:
                img3[i,j] = 1
    return img3


def foreground(img, img2, code = 0):

    h,w = img2.shape
    img = img.astype(int)
    img2 = img2.astype(int)
    img3 = np.zeros([h,w, 3], dtype=np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            if img2[i,j] == 1:
                img3[i,j, 0] = img[i,j, 0]
                img3[i,j, 1] = img[i,j, 1]
                img3[i,j, 2] = img[i,j, 2]
            else:
                if code == 1:
                    img3[i, j, 0] = img[i, j, 0] / 2
                    img3[i, j, 1] = img[i, j, 1] /2
                    img3[i, j, 2] = img[i, j, 2] / 2
                else:
                    img3[i,j] = 0
    return img3


def binary_smooth(img):  #using binary image
    h, w= img.shape
    for i in range(0,h-1):
        for j in range(0, w-1):
            if (img[i-1, j-1] == 0 and img[i-1, j] == 1 and img[i-1, j+1] ==0 and
             img[i, j-1] == 1 and img[i, j] == 1 and img[i, j+1] ==1 and
             img[i+1, j-1] == 0 and img[i+1, j] == 1 and img[i+1, j+1] ==0):
                img[i,j] = 1
    return img

def binary_dilate(img):
    h, w = img.shape
    out = img.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i,j] == 1:
                continue
            if (
                img[i-1,j-1] == 1 or img[i-1,j] == 1 or img[i-1,j+1] == 1 or
                img[i,  j-1] == 1 or img[i,  j+1] == 1 or
                img[i+1,j-1] == 1 or img[i+1,j] == 1 or img[i+1,j+1] == 1
            ):
                out[i,j] = 1

    return out


def binary_erode(img):
    h, w = img.shape
    out = img.copy()
    l = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            if (
                img[i-1,j-1] == 0 or img[i-1,j] == 0 or img[i-1,j+1] == 0 or
                img[i,  j-1] == 0 or img[i,  j] == 0 or img[i,  j+1] == 0 or
                img[i+1,j-1] == 0 or img[i+1,j] == 0 or img[i+1,j+1] == 0
            ):
                out[i,j] = 0
    return out


def background_remove(img, i):
    k = 0
    while k != i:
        img = binary_dilate(img)
        img = binary_erode(img)
        k += 1
    return img


def seed(img):
    h,w = img.shape
    img2 = np.zeros([h,w])
    for i in range(0, w-1):
        for j in range(0, h-1):
            if img[i,j] == 0:
                img2[i,j] = 0
            else :
                img2[i,j] = 1
    return img2
            

def connect_center(binary, seed):

    h,w = binary.shape
    while True:
        object = seed.copy()
        for i in range(1,h-2):
            for j in range(1,w-2):
                if seed[i,j] == 1:
                    for di in [0,1,-1]:
                        for dj in [0,1,-1]:
                            if binary[i+di,j+dj] == 1:
                                seed[i+di,j+dj] = 1
                
        if np.array_equal(object,seed):
            break

    return seed
            
                
    
                




