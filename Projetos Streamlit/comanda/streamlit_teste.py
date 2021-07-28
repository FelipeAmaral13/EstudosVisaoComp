# importar pacotes
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import imutils
import matplotlib.pyplot as plt
import os
from  glob import glob
import pytesseract
from pytesseract import Output
import pdf2image
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
import matplotlib.pyplot as plt
import time


path = os.getcwd()


#def deletar_txt(caminho):
#    '''Funcao para remover os arquivos png existentes.
#        entrada: caminho dos arquivos que serao removidos'''
#    filelist = [ f for f in os.listdir(caminho) if f.endswith(".png") ]
#    for f in filelist:
#        os.remove(os.path.join(caminho , f))
#
#deletar_txt(path +'\\imagesPDF')

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
			# st.write(type(our_image))
			st.image(our_image, width=300)
		

		texto = st.text_input("Escreva o texto aqui")
		st.write(f"{texto}")
		
				
		# Detecção Texto
		task = ["DetectTexto"]
		feature_choice = st.sidebar.selectbox("Detectar Texto",task)
		if st.button("Processar"):

			if feature_choice == 'DetectTexto':

				try:
					new_img = np.array(our_image.convert('RGB'))
				

					d = pytesseract.image_to_data(new_img, output_type=Output.DICT, lang='eng', config=tessdata_dir_config)
					n_boxes = len(d['level'])

					overlay = new_img.copy()
					for i in range(n_boxes):
						text = d['text'][i]
						if text == texto:
							(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
							(x1, y1, w1, h1) = (d['left'][i + 1], d['top'][i + 1], d['width'][i + 1], d['height'][i + 1])
							#(x2, y2, w2, h2) = (d['left'][i + 2], d['top'][i + 2], d['width'][i + 2], d['height'][i + 2])
							# cv2.rectangle(img, (x, y), (x1 + w1, y1 + h1), (0, 255, 0), 2)
							cv2.rectangle(overlay, (x, y), (x1 + w1, y1 + h1), (255, 0, 0), -1)
							#cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
							#cv2.rectangle(overlay, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), -1)
							#print(text)

				
					alpha = 0.4  # Transparency factor.
					# Following line overlays transparent rectangle over the image
					img_new = cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0)

					r = 1000.0 / img_new.shape[1]  # resizing image without loosing aspect ratio
					dim = (1000, int(img_new.shape[0] * r))
					# perform the actual resizing of the image and show it
					resized = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
					st.image(resized, width=300)
				except :
					st.info("Erro:    Selecione um arquivo válido!")
				
		



	elif choice == 'Sobre':
		st.subheader("Sobre API - Detecção Texto")
		st.markdown("Desenvolvido por Felipe Meganha")
		st.info("Contato: felipengmec@gmail.com.br")



if __name__ == '__main__':
    main()	