# importar pacotes
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import imutils
import matplotlib.pyplot as plt



def main():

    st.sidebar.header("API - Detect Plate Car")
    #st.sidebar.info("100% em Python")
    st.sidebar.markdown("API para detecção automática de placa de carro")

    # menu com oções de páginas
    opcoes_menu = ["Filtros", "Sobre"]
    escolha = st.sidebar.selectbox("Escolha uma opção", opcoes_menu)

    our_image = Image.open("Placa_teste.jpg")

    if escolha == "Filtros":
        st.title("API -  Detect Plate Car")
        st.text("por Felipe Meganha")
        st.markdown(f"""\n 
                    Objetivo: Detecção atumática da placa de carro. \n 
                    Retorno: ROI da placa do carro
                    \n""")

        # carregar e exibir imagem
        # our_image = cv2.imread(file_name)  ---> Não vai dar certo
        st.subheader("Carregar arquivo de imagem")
        image_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.sidebar.text("Imagem Original")
            st.sidebar.image(our_image, width=150)

        col1, col2 = st.beta_columns(2)
       

        # filtros que podem ser aplicados
        filtros = st.sidebar.radio("Filtros", ['Original', 'ROI - PLACA'])        

        if filtros == 'ROI - PLACA':

            # Converter para escala de cinza
            gray_image = cv2.cvtColor(np.array(our_image.convert('RGB')), cv2.COLOR_RGB2GRAY)

            # Aplicar filtro Bilateral 
            gray = cv2.bilateralFilter(gray_image, 13, 15, 15)

            # Morfologia - BlackHat
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 13))
            black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

            # Morfologia - Fechamento
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
            img_close = cv2.threshold(img_close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Deteccao de Bordas
            gradient_x = cv2.Sobel(black_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradient_x = np.absolute(gradient_x)

            # extrair valores minimos e máximos
            (minimo, maximo) = (np.min(gradient_x), np.max(gradient_x))

            # normalizar (valor - min) / (max - min)
            gradient_x = 255 * ((gradient_x - minimo) / (maximo - minimo))
            gradient_x = gradient_x.astype("uint8")

            # Aplicando filtros -> Suavizacao
            gradient_x = cv2.GaussianBlur(gradient_x, (13, 13), 0)
            gradient_x = cv2.morphologyEx(gradient_x, cv2.MORPH_CLOSE, kernel)
            thres = cv2.threshold(gradient_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            thres = cv2.erode(thres, None, iterations=3)  
            thres = cv2.dilate(thres, None, iterations=3)
            
            # Deteccao de contornos -> Placa
            contornos = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contornos = imutils.grab_contours(contornos)
            contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

            for c in contornos:
                (x, y, w, h) = cv2.boundingRect(c)
                proporcao = w / h

                # dimensoes da placa: 40x13cm
                if proporcao >= 3 and proporcao <= 5:
                    area_placa_identificada = gray[y: y + h, x: x + w]
                    placa_recortada = cv2.threshold(area_placa_identificada, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


            width = st.sidebar.slider("plot width", 200, 350, 200)
            
            with col1:
                col1.header("Original")
                col1.image(our_image, width=width)
            with col2:
                col2.header("ROI- PLACA")
                col2.image(area_placa_identificada, width=width)


            # st.image(gray_image, width=OUTPUT_WIDTH, caption="Imagem com filtro Grayscale")


        elif filtros == 'Original':
            width = st.sidebar.slider("plot width", 200, 500, 200)
            st.image(our_image, width=width)
        else:
            width = st.sidebar.slider("plot width", 200, 500, 200)
            st.image(our_image, width=width)

    elif escolha == 'Sobre':
        st.subheader("POC de uma API para detecção automática da posição da placa de carro.")
        st.markdown("""A posição da imagem do carro é importante pois a filtragem pode variar de imagem para imagem. 
                    Foi priorizado uma imagem que simula uma Cancela de estacionamento.""")
        st.text("Felipe Meganha")
        

if __name__ == '__main__':
    main()
