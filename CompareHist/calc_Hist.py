import glob
import cv2
import os
import pandas as pd


class CompareImages:
    def __init__(self):
        self.index = {}
        self.images = {}
        self.OPENCV_METHODS = (
            ("Correlation", cv2.HISTCMP_CORREL),
            ("Chi-Squared", cv2.HISTCMP_CHISQR),
            ("Intersection", cv2.HISTCMP_INTERSECT),
            ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA))

    def _extract_histogram(self, image):
        hist = cv2.calcHist(
            [image], [0, 1], None, [8, 8], [0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _load_images(self):
        for imagePath in glob.glob(os.getcwd() + "\\*.jpg"):
            filename = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            self.images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hist = self._extract_histogram(image)
            self.index[filename] = hist

    def compare(self, imagem_analisada):
        self._load_images()

        lista_resultados = []
        lista_methodName = []

        for (methodName, method) in self.OPENCV_METHODS:
            results = {}
            reverse = False

            if methodName in ("Correlation", "Intersection"):
                reverse = True

            for (k, hist) in self.index.items():
                d = cv2.compareHist(
                    self.index[os.getcwd() + imagem_analisada], hist, method)
                results[k] = d

            lista_methodName.append(methodName)
            results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
            lista_resultados.append(pd.DataFrame(results, columns=['Distância', 'Arquivo']))

        df = pd.concat(lista_resultados)
        df['Metodo'] = lista_methodName
        df.to_csv('Resultado_compareHist.csv', sep=';', encoding='latin1', index=False)

        return df


comparador = CompareImages()
comparador.compare(r"C:\Users\felip\OneDrive\Desktop\Github Meganha\EstudosVisaoComp\CompareHist\Images\apple1.jpg")