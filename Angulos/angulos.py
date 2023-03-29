import cv2
import math
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Calculates angle between lines in an image')
    parser.add_argument('--image_path', type=str, help='Path to image file')
    return parser.parse_args()


def validate_image_path(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f'Image file "{image_path}" not found')


def mouse_points(event, x, y, flags, params):
    '''
    Função para captação dos clicks do mouse.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(params['points_list'])
        if size != 0 and size % 3 != 0:
            cv2.line(
                params['img'],
                tuple(params['points_list'][round((size-1)/3)*3]), (x, y), (0, 255, 0), 2)
        cv2.circle(params['img'], (x, y), 5, (0, 0, 255), cv2.FILLED)
        params['points_list'].append([x, y])


def slope(pt1, pt2):  # Slope das retas
    '''
    Função para pegar o slope, ou também conhecido como gradiente, das retas
    '''
    return (pt2[1] - pt1[1])/(pt2[0] - pt1[0])


def get_angle(points_list, img):
    '''
    Função para pegar o angulo a partir de três pontos
    '''
    if len(points_list) < 3:
        print('Not enough points selected')
        return

    pt1, pt2, pt3 = points_list[-3:]
    m1 = slope(pt1, pt2)
    m2 = slope(pt1, pt3)
    angR = math.atan((m2-m1)/(1 + (m2*m1)))  # Angulos em Radianos
    angD = round(math.degrees(angR))  # Angulos em Grau
    if angD < 0:
        angD += 360

    cv2.putText(img, str(angD), tuple(pt1),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)


def main(image_path):
    validate_image_path(image_path)

    img = cv2.imread(image_path)
    img_copy = img.copy()
    points_list = []

    while True:
        if len(points_list) % 3 == 0 and len(points_list) != 0:
            get_angle(points_list, img)

        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', mouse_points, {'img': img, 'points_list': points_list})

        # Reset da imagem. Apagar todos os textos inseridos na imagem
        if cv2.waitKey(1) == ord('n'):
            img = img_copy.copy()
            points_list = []
            cv2.imshow('Image', img)

        # Apertar 'ESC' para sair
        if cv2.waitKey(1) == 27:
            points_list = []
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    args = parse_args()
    main(args.image_path)
