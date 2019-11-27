from skimage.exposure import rescale_intensity
import numpy as np


class Threshold:

    @staticmethod
    def run(image, t=45):
        (iH, iW) = image.shape[:2]
        output = np.zeros((iH, iW), dtype="float32")

        for h in range(0, iH):
            for w in range(0, iW):
                intensity = image[h, w]
                print(intensity)
                if intensity <= t:
                    x = 0
                else:
                    x = 255
                output[h, w] = x

        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        return output