from threshold import Threshold as th
from rgb_to_gray import RgbToGray
from convolution import Convolution
import cv2
from skimage.morphology import erosion, dilation
from erosion import Erosion
from dilation import Dilation

img = cv2.imread('3d_pokemon.png')

gray_img = RgbToGray.run1(img)
gray_img = Convolution(gray_img, kernel_name='gaussian5').run()
binary_img = th.run(gray_img)

# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# binary_img = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

# eroded_img = cv2.erode(binary_img, None, iterations=2)
# eroded_img = erosion(binary_img)
# eroded_img_2 = Erosion(eroded_img).run()
dilated_img = dilation(binary_img)
dilated_img_2 = Dilation(binary_img).run()

cv2.imshow('dilated image', dilated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()