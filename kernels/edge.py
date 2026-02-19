import numpy as np

# 1. Sobel X (vertical edges)
kernel_sobel_x = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])


# 2. Sobel Y (horizontal edges)
kernel_sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])


# 3. Prewitt X
kernel_prewitt_x = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
])


# 4. Prewitt Y
kernel_prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])


# 5. Roberts Cross (X)
kernel_roberts_x = np.array([
    [ 1,  0],
    [ 0, -1]
])


# 6. Roberts Cross (Y)
kernel_roberts_y = np.array([
    [ 0,  1],
    [-1,  0]
])


# 7. Laplacian (edge-only)
kernel_laplacian_edge = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
])


# 8. Laplacian (strong edge)
kernel_laplacian_edge_strong = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
