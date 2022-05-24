import numpy as np
import scipy


def find_endpoints(img):
    kernel = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    img_conv = scipy.signal.convolve2d(
        img.astype(np.int32),
        kernel, mode='same'
    )
    endpoints = np.stack(
        np.where((img == 255) & (img_conv == 255)),
        axis=1
    )
    return endpoints
