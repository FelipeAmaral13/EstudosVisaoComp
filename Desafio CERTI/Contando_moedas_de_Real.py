#!/usr/local/bin/python

# Libraries
import cv2
import numpy as np
import os


# Check if the "image_results" directory exists
if os.path.isdir('image_result') == False:
    print('The "image_result" directory does not exist. Creating directory.')
    os.mkdir('image_result')
else:
    print('The "image_result" directory does exist.')


# Capturing local path
path = os.getcwd()


def count_real():

    '''
    This function has the objective of detecting, pointing out and accounting the amount of coins.

    Returns:
        Coins detected.
    '''
        
    # Read the original image
    img_real_coin = cv2.imread('real_original.jpg')

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

    # Apply the Blur Filter
    # Ksize = (19,19). Better value compared to the proposed
    img_Blur = cv2.blur(img_Gray, (19, 19))
    cv2.imshow('Blurred Image', img_Blur)
    cv2.waitKey(0)

    # Detect circles with HoughCircles
    circles = cv2.HoughCircles(
        img_Gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=50, minRadius=15, maxRadius=70)

    # Checking if a circle is found
    if circles is not None:
        circles = np.uint16(np.around(circles))

        print(f'Number of coins detected = {circles.shape[1]}')

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img_real_coin, (i[0], i[1]), i[2], (0, 255, 0), 3)
            # draw the center of the circle
            cv2.circle(img_real_coin, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print(f'Cant found circles')

    cv2.imshow('detected circles', img_real_coin)
    cv2.waitKey(0)

    # Save the image in Folder
    print(f'Your image will be saved on the folder {path}\\image_result')
    cv2.imwrite(path + '\\image_result\\real_result.jpg', img_real_coin)



if __name__ == "__main__":

    count_real()