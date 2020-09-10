#Bibliotecas
import cv2
import numpy as np

#Imagem teste
image = cv2.imread('lena.png')
image_to_show = np.copy(image)

#Estados iniciais do mouse
mouse_pressed = False
x_init = y_init = top_left_pt = bottom_right_pt = -1


def mouse_callback(event, x, y, flags, param):
    global image_to_show, x_init, y_init, top_left_pt, bottom_right_pt, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x_init, y_init = x, y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            cv2.rectangle(image_to_show, (x_init, y_init),
                          (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        top_left_pt, bottom_right_pt = x, y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', image_to_show)
    k = cv2.waitKey(1)

    if k == ord('c'):
        if y_init > bottom_right_pt:
            y_init, bottom_right_pt = bottom_right_pt, y_init
        if x_init > top_left_pt:
            x_init, top_left_pt = top_left_pt, x_init

        if bottom_right_pt - y_init > 1 and top_left_pt - x_init > 0:
            image = image[y_init:bottom_right_pt, x_init:top_left_pt]
            image_to_show = np.copy(image)
    elif k == 27:
        break

cv2.destroyAllWindows()