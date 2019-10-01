import numpy as np

def center(arr, view_r, view_c):
    arr_shape = arr.shape

    p_c = view_c // 2
    p_r = view_r // 2

    pad_shape = (arr_shape[0] + 2 * p_r, arr_shape[1] + 2 * p_c)

    pad = np.zeros(shape=pad_shape, dtype=int)
    pad[:p_c, :]  = 8 # Top
    pad[-p_c:, :] = 8 # Bottom
    pad[:, :p_c]  = 4 # Left
    pad[:, -p_c:] = 4 # Right

    pad[p_r: p_r + arr_shape[0], p_c: p_c + arr_shape[1]] = arr

    return pad

arr = np.zeros(shape=(3, 8)) + 1

center(arr, 11, 11)