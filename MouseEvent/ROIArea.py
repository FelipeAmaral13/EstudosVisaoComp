import cv2
from matplotlib import pyplot as plt
import numpy as np

def sketch_transform(image):

    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 100, 150)
    #_, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return image_canny

cam_capture = cv2.VideoCapture('1.mp4')


upper_left = (520, 520)
bottom_right = (820, 820)


while True:
    ret, image_frame = cam_capture.read()
    
    #ROI
    r = cv2.rectangle(image_frame, upper_left, bottom_right, (255, 0, 0), 2)

    rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
    sketcher_rect = rect_img
    sketcher_rect = sketch_transform(sketcher_rect)
    
    #Conversao dos 3 canais para voltar a imagem original
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    
    #Substituindo a imagem esboçada na região de interesse
    image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb


    cv2.imshow("Sketcher ROI", image_frame)
    if cv2.waitKey(1) == 27:
        break
        
cam_capture.release()
cv2.destroyAllWindows()