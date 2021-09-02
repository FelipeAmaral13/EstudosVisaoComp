# importar pacotes
import streamlit as st
import cv2
from PIL import Image
import os
import pytesseract
from pytesseract import Output
import numpy as np


path = os.getcwd()

# Tesseract
pytesseract.pytesseract.tesseract_cmd = path + r'\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = f'--tessdata-dir "{path}\\Tesseract-OCR\\tessdata"'


def main():

	global path
	
	# Titulo da API
	st.title("API - Comanda Detect Text")

	# Opcoes de tarefas da API
	activities = ["DetectText", "Sobre"] 
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'DetectText':
		st.subheader("Detecção Texto")

		# Upload do arquivo
		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Original Image")
			st.image(our_image, width=600)		

        # Texto para busca
		texto = st.text_input("Escreva o texto aqui")
		st.write(f"{texto}")
		
				
		# Detecção Texto
		task = ["DetectTexto"]
		feature_choice = st.sidebar.selectbox("Detectar Texto",task)
		if st.button("Processar"):

			# Se palavra encontrada
			if feature_choice == 'DetectTexto':

				try:
					new_img = np.array(our_image.convert('RGB'))			

					d = pytesseract.image_to_data(new_img, output_type=Output.DICT, lang='por', config=tessdata_dir_config)
					n_boxes = len(d['level'])
					overlay = new_img.copy()
					for i in range(n_boxes):
						text = d['text'][i]
						print(text)						
						if text == texto:
							(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
							print(d['left'][i], d['top'][i], d['width'][i], d['height'][i])
							(x1, y1, w1, h1) = (d['left'][i + 1], d['top'][i + 1], d['width'][i + 1], d['height'][i + 1])
							cv2.rectangle(overlay, (x, y), (x1 + w1, y1 + h1), (255, 0, 0), -1)
				
					alpha = 0.4  # Fator para transparência
					# A linha seguinte sobrepõe o retângulo transparente sobre a imagem
					img_new = cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0)

					r = 1000.0 / img_new.shape[1]  # resizing da imagem sem perder o ratio calculado
					dim = (1000, int(img_new.shape[0] * r))
					# realizar o redimensionamento real da imagem e mostrá-la
					resized = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
					st.image(resized, width=600)
				except :
					st.info("Erro:    Selecione um arquivo válido!")
				
		



	elif choice == 'Sobre':
		st.subheader("Sobre API - Detecção Texto")
		st.markdown("Desenvolvido por Felipe Meganha")
		st.info("Contato: felipengmec@gmail.com.br")



if __name__ == '__main__':
	main()	