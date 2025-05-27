import cv2
import numpy as np
from utils.file_utils import load_image, save_image, ensure_dir
from utils.image_utils import show_image
import os

def process_dollar(image_path, output_dir='image_result'):
    ensure_dir(output_dir)
    img = load_image(image_path)
    show_image('Original', img)

    b, g, r = cv2.split(img)

    show_image('Canal G', g)

    
    ret, imgBinary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image("imgBinary", imgBinary)

    struct_element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    imgdilate = cv2.dilate(imgBinary, struct_element1, iterations=1)
    imgerode = cv2.erode(imgdilate, struct_element1, iterations=1)
    show_image("Imagem Dilatada", imgerode)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minArea = 800
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.maxCircularity = 1.0
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(imgerode)

    for kp in keypoints:
        cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/2), (0, 255, 0), 2)

    print(f'Number of coins detected = {len(keypoints)}')

    res = cv2.drawKeypoints(imgerode, keypoints, np.array([]), (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    show_image("img", img)


