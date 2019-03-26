import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2

# sourceTxt = "sourceToto.png"
# modifTxt = "toto1.png"
sourceTxt = "IAsource.jpg"
modifTxt = "IAv1.jpg"

imgColor = cv2.IMREAD_COLOR

## Rad images (Attention BGR color space)
img1 = cv2.imread(sourceTxt,imgColor)
img2 = cv2.imread(modifTxt,imgColor)

## Convert to RGB color space (Matplotlib)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

diff = cv2.absdiff(img1, img2) # Calculates the per-element absolute difference between two images arrays
# mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
# cv2.imwrite("mask", mask)

th = 15
imask =  mask>th

canvas = np.zeros_like(img2, np.uint8)
canvas[imask] = img2[imask]

imgResult = cv2.addWeighted(img2, 0.5, canvas, 1, 0) 

resultTxt = 'diff__'+sourceTxt+'__'+modifTxt

## Convert back blended image to BGR
img_converted = cv2.cvtColor(imgResult, cv2.COLOR_RGB2BGR)
cv2.imwrite(resultTxt, img_converted)
# cv2.imwrite("difference",diff)

images=[]
title_images=[]

images.append(img1),title_images.append(sourceTxt)
images.append(img2),title_images.append(modifTxt)
images.append(diff),title_images.append("difference")
images.append(mask),title_images.append("mask")
images.append(canvas),title_images.append("canvas")
images.append(imgResult),title_images.append(resultTxt)

for i in xrange(len(images)):
    plt.subplot(2,3,i+1),plt.imshow(images[i])
    plt.title(title_images[i])
    plt.xticks([]),plt.yticks([])

plt.show()