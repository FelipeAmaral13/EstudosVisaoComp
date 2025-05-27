import cv2
import numpy as np
from utils.file_utils import load_image, save_image, ensure_dir
from utils.image_utils import show_image
import os

def process_real(image_path, output_dir='image_result'):
    """Process real coins using HoughCircles."""
    ensure_dir(output_dir)
    img = load_image(image_path)
    show_image('Original', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image('Gray', gray)

    blurred = cv2.blur(gray, (19, 19))
    show_image('Blurred', blurred)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=50, minRadius=15, maxRadius=70)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f'Number of coins detected = {circles.shape[1]}')
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print('No circles found.')

    show_image('Detected Circles', img)
    save_image(os.path.join(output_dir, 'real_result.jpg'), img)
