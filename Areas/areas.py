import cv2
import numpy as np
from shapely.geometry import Polygon


class ShapeCalculator:
    def __init__(self):
        # Definir constantes
        self.IMAGE_SIZE = (800, 800)
        self.BACKGROUND_COLOR = (0, 0, 0)
        self.RECTANGLE_COLOR = (0, 255, 0)
        self.ELLIPSE_COLOR = (255, 0, 0)
        self.POINT_COLOR = (0, 0, 255)
        self.TEXT_COLOR = (255, 0, 0)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 1
        self.FONT_THICKNESS = 2
        self.background = None
        self.pointsList = []

    def create_background(self):
        self.background = np.zeros((self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3), dtype='uint8')
        self.background[:] = self.BACKGROUND_COLOR

    def create_shapes(self):
        rectangle1 = cv2.rectangle(self.background, (50, 50), ((self.background.shape[1]//2)+50, (self.background.shape[0]//2)+50), self.RECTANGLE_COLOR, thickness=-1)
        rectangle2 = cv2.rectangle(self.background, (350, 500), ((self.background.shape[1]//4)+350,(self.background.shape[0]//4)+350), self.RECTANGLE_COLOR, thickness=-1)
        ellipse = cv2.ellipse(self.background, (600, 200), (100, 50), 0, 0, 180, self.ELLIPSE_COLOR, -1)

    def mouse_points(self, event, x, y, flags, params):
        '''
        Função para captação dos clicks do mouse.
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.background, (x, y), 5, self.POINT_COLOR, cv2.FILLED)
            self.pointsList.append([x, y])
            print(self.pointsList)

    def calculate_area(self):
        if len(self.pointsList) >= 3:
            area1 = np.array(self.pointsList)
            cv2.fillPoly(self.background, [area1], (255, 255, 255))

            polygon = Polygon(self.pointsList)
            area = polygon.area
            perimeter = polygon.length

            cv2.putText(self.background, f"Area: {area:.2f}, Perimeter: {perimeter:.2f}", (50, 700), self.FONT, self.FONT_SCALE, self.TEXT_COLOR, self.FONT_THICKNESS)
        else:
            print("Selecione pelo menos 3 pontos antes de calcular a área")

    def run(self):
        self.create_background()
        self.create_shapes()

        cv2.imshow('Image', self.background)
        cv2.setMouseCallback('Image', self.mouse_points)

        while True:
            cv2.imshow('Image', self.background)

            # Reset da imagem. Apagar todos os textos inseridos na imagem
            if cv2.waitKey(1) == ord('n'):
                self.create_background()
                self.pointsList = []
                cv2.imshow('Image', self.background)

            # Aperte M para calculo da area
            if cv2.waitKey(1) == ord('m'):
                self.calculate_area()

            # Apertar 'ESC' para sair
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    sc = ShapeCalculator()
    sc.run()