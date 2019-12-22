from skimage.exposure import rescale_intensity
import numpy as np
import dtype


class RgbToGray:
    def __init__(self, img):
        self.img = img

    def run1(self):
        arr = np.asanyarray(self.img[..., :3])
        rgb = dtype.img_as_float(arr)
        coeffs = np.array([0.3, 0.59, 0.11], dtype=rgb.dtype)
        output = rgb @ coeffs
        output = rescale_intensity(output, in_range=(0, 1))
        output = (output * 255).astype("uint8")

        return output

    def run2(self):
        arr = np.asanyarray(self.img[..., :3])
        rgb = dtype.img_as_float(arr)
        coeffs = np.array([1/3, 1/3, 1/3], dtype=rgb.dtype)
        output = rgb @ coeffs
        output = rescale_intensity(output, in_range=(0, 1))
        output = (output * 255).astype("uint8")

        return output