#!/usr/local/bin/python

# Libraries
import cv2
import numpy as np
import os


# Check if the "image_results" directory exists
if os.path.isdir('image_result') is False:
    print('The "image_result" directory does not exist. Creating directory.')
    os.mkdir('image_result')
else:
    print('The "image_result" directory does exist.')


# Capturing local path
path = os.getcwd()


def count_dolar():

    '''
    This function has the objective of detecting,
    pointing out and counting the amount of coins.

    Returns:
        Coins detected.
    '''

    # Read image
    img_dolar_coin = cv2.imread(os.path.join('Images', 'dolar_original.png'))

    # Verify if can read the image
    if img_dolar_coin is not None:
        cv2.imshow('Original Image', img_dolar_coin)
        cv2.waitKey(0)
    else:
        print('Cant read image. Please make sure the image is in the folder ')

    # convert the image to gray scale
    img_Gray = cv2.cvtColor(img_dolar_coin, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Scale Image', img_Gray)
    cv2.waitKey(0)

    # Bilateral Filter
    img_Blur = cv2.bilateralFilter(img_Gray, 1, 75, 75)

    # Binary Image
    bin_type = cv2.THRESH_BINARY_INV
    _, imgBinary = cv2.threshold(img_Blur, 47, 255, bin_type)
    cv2.imshow('Binary Image', imgBinary)
    cv2.waitKey(0)

    # Morphological Filters
    struct_element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    imgdilate = cv2.dilate(imgBinary, struct_element1, iterations=7)
    cv2.imshow('Morphologic Image', imgdilate)
    cv2.waitKey(0)

    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 0
    params.maxThreshold = 255
    params.blobColor = 0
    params.minArea = 45
    params.maxArea = 50000
    params.filterByCircularity = True
    params.filterByConvexity = False
    params.minCircularity = 0.6
    params.maxCircularity = 1

    det = cv2.SimpleBlobDetector_create(params)
    keypts = det.detect(imgdilate)

    res = cv2.drawKeypoints(
        imgdilate, keypts, np.array([]),
        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    coin = 0
    for kp in keypts:
        # print("Centroide: (%f,%f)"%(kp.pt[0],kp.pt[1]))
        coin += 1
        cv2.rectangle(img_dolar_coin, (int(kp.pt[0]), int(
            kp.pt[1])), (int(kp.pt[0])+1, int(kp.pt[1])+1), (255, 0, 0), 8)
        cv2.circle(img_dolar_coin, (int(kp.pt[0]), int(
            kp.pt[1])), int(kp.size/2), (0, 255, 0), 2)

    print(f'Number of coins detected = {coin}')

    cv2.imshow("SimpleBlobDetector Image", res)
    cv2.waitKey(0)

    cv2.imshow("SimpleBlobDetector Results", img_dolar_coin)
    cv2.waitKey(0)

    # Save the image in Folder
    print(f'Your image will be saved on the folder {path}\\image_result')
    cv2.imwrite(
        os.path.join(
            path, 'Project', 'image_result', 'dolar_result.jpg', img_dolar_coin
            ))


if __name__ == "__main__":

    count_dolar()
