#!/usr/bin/python

import sys
import numpy as np
import cv2
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import cv2
import copy

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

# Iitialisation
sourceTxt = str(sys.argv[1])
modifTxt = str(sys.argv[2])
th = 10 # threshold value used when mask is calculted (default value)
if (len(sys.argv)>3):
    th = int(sys.argv[3])

# Display parameter
print("--- "+str(sys.argv[0])+" ---")
print("     source image : "+sourceTxt)
print("     modified image : "+modifTxt)
print("     threshold : "+str(th))
print("------")


imgColor = cv2.IMREAD_COLOR # Set flag to tell openCV to read image in color

## Read images (Attention BGR color space)
imgSource = cv2.imread(sourceTxt,imgColor)
imgModified = cv2.imread(modifTxt,imgColor)

## Convert to RGB color space (Matplotlib)
imgSource = cv2.cvtColor(imgSource, cv2.COLOR_BGR2RGB)
imgModified = cv2.cvtColor(imgModified, cv2.COLOR_BGR2RGB)

## Compute difference between imgSource and imgModified
diff = cv2.absdiff(imgSource, imgModified) # Calculates the per-element absolute difference between two images arrays
mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) # Seem to work uncorrectly

imask =  mask>th # Create mask based on threshold value

canvas = np.zeros_like(imgModified, np.uint8)
canvas[imask] = imgModified[imask]

imgResult = copy.deepcopy(canvas) # Duplicate image canvas (deep copy)
imgResult[:,0:imgResult.size] = (0,0,0)  # (B, G, R) # Reset canvas to black image
imgResult = cv2.addWeighted(imgSource, 0.5, imgResult, 1, 0) # Blend imgSource on black canvas with alpha value
imgResult[imask] = canvas[imask] # Hilight modified pixels

## Save blended image

resultTxt = 'diff__'+sourceTxt+'__'+modifTxt
## Convert back blended image to BGR
img_converted = cv2.cvtColor(imgResult, cv2.COLOR_RGB2BGR)
cv2.imwrite(resultTxt, img_converted) # Save image "img_converted" with name "resultTxt"


## Dispaly images

images=[] # List images
title_images=[] # List images titles

nb_row = 3
nb_column =2

# Add images elements to list images and title_images
# -- Warning : Add as much as element as defined by nb_row*nb_column --

images.append(imgSource),title_images.append(sourceTxt)
images.append(imgModified),title_images.append(modifTxt)
images.append(diff),title_images.append("difference")
images.append(mask),title_images.append("mask (grayscale)")
images.append(canvas),title_images.append("canvas  threshold:"+str(th))
images.append(imgResult),title_images.append(resultTxt)

f, ax = plt.subplots(nb_column, nb_row, sharex=True, sharey=True) # Create a set of subplot with shared x and y axis
for i in xrange(len(images)):
    idx_row = i/nb_row
    idx_col = (i%nb_row)
    print("case : ["+str(idx_row)+","+str(idx_col)+"]")
    ax[idx_row,idx_col].imshow(images[i])
    ax[idx_row,idx_col].set_title(title_images[i])

plt.xticks([]),plt.yticks([]) # Disable scale marking on x and w axis

plt.show() # Display subplot