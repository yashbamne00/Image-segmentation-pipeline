# src/filters.py
import numpy as np
    
#sharpning
kernel_laplacian_4 = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

kernel_high_boost = np.array([
    [-1, -1, -1],
    [-1, 10, -1],
    [-1, -1, -1]
])


kernel_horizontal = np.array([
    [ 0,  0,  0],
    [-1,  2, -1],
    [ 0,  0,  0]
])

kernel_vertical = np.array([
    [ 0, -1,  0],
    [ 0,  2,  0],
    [ 0, -1,  0]
])
