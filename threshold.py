from skimage.exposure import rescale_intensity
import numpy as np


class Threshold:
    def __init__(self, img, th=45):
        self.img = img
        self.th = th

    def run(self):
        (iH, iW) = self.img.shape[:2]
        output = np.zeros((iH, iW), dtype="float32")

        for h in range(0, iH):
            for w in range(0, iW):
                intensity = self.img[h, w]
                if intensity <= self.th:
                    x = 0
                else:
                    x = 255
                output[h, w] = x

        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        return output