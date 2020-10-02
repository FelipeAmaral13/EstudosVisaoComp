#!/usr/local/bin/python

# Libraries
import cv2
import numpy as np
import os

img_real_coin = cv2.imread('dolar_original.png')

# Verify if can read the image
if not img_real_coin is None:
    cv2.imshow('Original Image', img_real_coin)
    cv2.waitKey(0)
else:
    print('Cant read image. Please make sure the image is in the folder ')


# convert the image to gray scale
img_Gray = cv2.cvtColor(img_real_coin, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Scale Image', img_Gray)
cv2.waitKey(0)


img_Blur = cv2.bilateralFilter(img_Gray, 1, 75, 75)
cv2.imshow('Blurred Image', img_Blur)
cv2.waitKey(0)


# Binary Image
bin_type = cv2.THRESH_BINARY_INV  
limThreshold, imgBinary = cv2.threshold(img_Blur, 47, 255, bin_type)
cv2.imshow('Binary Image', imgBinary)
cv2.waitKey(0)


# Morphological Filters
struct_element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
imgdilate = cv2.dilate(imgBinary, struct_element1, iterations=7)
struct_element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
imag_erode = cv2.morphologyEx(imgdilate, cv2.MORPH_OPEN, struct_element2, iterations=4)
cv2.imshow('Morphologic Image', imgdilate)
cv2.waitKey(0)


# blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.minThreshold = 0
params.maxThreshold = 255
params.blobColor = 0
params.minArea = 1
params.maxArea = 50000
params.filterByCircularity = True
params.filterByConvexity = False
params.minCircularity =.4
params.maxCircularity = 1

det = cv2.SimpleBlobDetector_create(params)
keypts = det.detect(imgdilate)

im_with_keypoints = cv2.drawKeypoints(imgdilate, keypts, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
res = cv2.drawKeypoints(imgdilate, keypts, np.array([]), (0, 0, 255 ), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


#cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("RES", res)
cv2.waitKey(0)