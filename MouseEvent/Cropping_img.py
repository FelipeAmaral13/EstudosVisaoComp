#Bibliotecas
import cv2
import numpy as np

#Imagem teste
image = cv2.imread('lena.png')
image_to_show = np.copy(image)

#Estados iniciais do mouse
cropping = False
x_init, y_init, top_left_pt, bottom_right_pt = 0, 0, 0, 0


def mouse_callback(event, x, y, flags, param):
    global image_to_show, x_init, y_init, top_left_pt, bottom_right_pt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        x_init, y_init = x, y
        image_to_show = np.copy(image)
        print(x_init)
        print(y_init)

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            image_to_show = np.copy(image)
            cv2.rectangle(image_to_show, (x_init, y_init),
                          (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        top_left_pt, bottom_right_pt = x, y
        print(top_left_pt)
        print(bottom_right_pt)




cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:

    cv2.imshow('image', image_to_show)
    k = cv2.waitKey(1)

    if k == ord('r'):
        if y_init > bottom_right_pt:
            y_init, bottom_right_pt = bottom_right_pt, y_init
        if x_init > top_left_pt:
            x_init, top_left_pt = top_left_pt, x_init

        if bottom_right_pt - y_init > 1 and top_left_pt - x_init > 0:
            image = image[y_init:bottom_right_pt, x_init:top_left_pt]
            print(image)
            image_to_show = np.copy(image)

    if k == ord('s'):
        cv2.imwrite('teste.jpg', image_to_show)

    if cv2.waitKey(1) == ord('q'):
        break
 
cv2.destroyAllWindows()