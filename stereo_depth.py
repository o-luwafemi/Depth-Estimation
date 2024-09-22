import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
# Open left and right images
imgL = cv.imread('./images/image_l.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('./images/image_r.png', cv.IMREAD_GRAYSCALE)



# Block size makes the results smoother, but produce less accurate disparity map.
stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)

disparity = stereo.compute(imgL,imgR)


plt.imshow(disparity,'gray')
plt.show()