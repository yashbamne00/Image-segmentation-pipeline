import numpy as np

# 1. Mean / Box blur (3x3)
kernel_mean_3 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9


# 3. Gaussian blur (3x3)
kernel_gaussian_3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16



# 5. Light smoothing (center-weighted)
kernel_light_smooth = np.array([
    [0, 1, 0],
    [1, 4, 1],
    [0, 1, 0]
]) / 8


# 6. Strong smoothing
kernel_strong_smooth = np.array([
    [1, 1, 1],
    [1, 8, 1],
    [1, 1, 1]
]) / 16


# 6. binary smoothing
binary_smooth = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
]) / 16

