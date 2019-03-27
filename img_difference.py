import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2
import copy

# sourceTxt = "sourceToto.png"
# modifTxt = "toto1.png"
sourceTxt = "IAsource.jpg"
modifTxt = "IAv1.jpg"

imgColor = cv2.IMREAD_COLOR

## Rad images (Attention BGR color space)
imgSource = cv2.imread(sourceTxt,imgColor)
imgModified = cv2.imread(modifTxt,imgColor)

## Convert to RGB color space (Matplotlib)
imgSource = cv2.cvtColor(imgSource, cv2.COLOR_BGR2RGB)
imgModified = cv2.cvtColor(imgModified, cv2.COLOR_BGR2RGB)

diff = cv2.absdiff(imgSource, imgModified) # Calculates the per-element absolute difference between two images arrays
# mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
# cv2.imwrite("mask", mask)

th = 10
imask =  mask>th

canvas = np.zeros_like(imgModified, np.uint8)
canvas[imask] = imgModified[imask]

imgResult = copy.deepcopy(canvas) # Duplicate image canvas 
imgResult[:,0:imgResult.size] = (0,0,0)      # (B, G, R) # Reset canvas to black image
imgResult = cv2.addWeighted(imgSource, 0.5, imgResult, 1, 0) # Blend imgSource on black canvas with alpha value
imgResult[imask] = canvas[imask] # Hilight modified pixels


resultTxt = 'diff__'+sourceTxt+'__'+modifTxt

## Convert back blended image to BGR
img_converted = cv2.cvtColor(imgResult, cv2.COLOR_RGB2BGR)
cv2.imwrite(resultTxt, img_converted)
# cv2.imwrite("difference",diff)

images=[]
title_images=[]

images.append(imgSource),title_images.append(sourceTxt)
images.append(imgModified),title_images.append(modifTxt)
images.append(diff),title_images.append("difference")
images.append(mask),title_images.append("mask (grayscale)")
images.append(canvas),title_images.append("canvas  threshold:"+str(th))
images.append(imgResult),title_images.append(resultTxt)

for i in xrange(len(images)):
    plt.subplot(2,3,i+1),plt.imshow(images[i])
    plt.title(title_images[i])
    plt.xticks([]),plt.yticks([])

plt.show()