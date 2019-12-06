from skimage.exposure import rescale_intensity
import numpy as np
import cv2

class Convolution:
    def __init__(self, img, kernel_name='sharpen'):
        self.img = img
        self.kernel_name = kernel_name

    def initialize(self):
        sharpen = np.array((
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]), dtype="int")

        laplacian = np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype="int")

        sobelX = np.array((
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]), dtype="int")

        sobelY = np.array((
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]), dtype="int")

        gaussian3 = np.array((
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]), dtype="float") * (1.0/16.0)

        gaussian5 = np.array((
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]), dtype="float") * (1.0 / 256.0)

        smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
        largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

        self.kernelBank = {
            "small_blur": smallBlur,
            "large_blur": largeBlur,
            "sharpen": sharpen,
            "laplacian": laplacian,
            "sobel_x": sobelX,
            "sobel_y": sobelY,
            "gaussian3": gaussian3,
            "gaussian5": gaussian5
        }

    def run(self):
        self.initialize()

        (iH, iW) = self.img.shape[:2]
        (kH, kW) = self.kernelBank[self.kernel_name].shape[:2]
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(self.img, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                k = (roi * self.kernelBank[self.kernel_name]).sum()
                output[y - pad, x - pad] = k

        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        return output