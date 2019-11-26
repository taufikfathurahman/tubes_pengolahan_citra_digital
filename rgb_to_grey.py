from skimage import io
import numpy as np
import dtype


class RgbToGrey:
    def __init__(self, img):
        self.img = img

    def run1(self):
        # prepare color array
        arr = np.asanyarray(self.img[..., :3])
        rgb = dtype.img_as_float(arr)
        coeffs = np.array([0.3, 0.59, 0.11], dtype=rgb.dtype)

        return rgb @ coeffs

    def run2(self):
        # prepare color array
        arr = np.asanyarray(self.img[..., :3])
        rgb = dtype.img_as_float(arr)
        coeffs = np.array([1/3, 1/3, 1/3], dtype=rgb.dtype)

        return rgb @ coeffs