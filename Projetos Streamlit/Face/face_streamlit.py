import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detect_faces(our_image):

	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 4)

	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
            			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

	return img,faces 

def detect_eyes(our_image):

	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detect_smiles(our_image):

	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	
    # Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.5, 6)

	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img   

def main():

	st.title("API - Face Detection")
	st.text("Desenvolvido em Python e Streamlit")

	activities = ["Deteccao","Sobre"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Deteccao':
		st.subheader("Face Detection")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        
		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Original Image")
			# st.write(type(our_image))
			st.image(our_image)

		col1, col2 = st.beta_columns(2)
		
		# Face Detection
		task = ["Faces","Sorriso","Olhos",]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):

			if feature_choice == 'Faces':
				try:
					result_img,result_faces = detect_faces(our_image)
					st.image(result_img)
					st.success(f"Faces Encontradas: {len(result_faces)}")
					
				except:
					st.info("Error: Nenhuma imagem selecionada")

			elif feature_choice == 'Sorriso':

				try:
					result_img = detect_smiles(our_image)
					st.image(result_img)
				except:
					st.info("Error: Nenhuma imagem selecionada")


			elif feature_choice == 'Olhos':
				try:
					result_img = detect_eyes(our_image)
					st.image(result_img)
				except:
					st.info("Error: Nenhuma imagem selecionada")

	elif choice == 'Sobre':
		st.subheader("Sobre API - Face Detection")
		st.markdown("Desenvolvido por [FelipeMeganha](https://www.udemy.com/course/visao-computacional-com-python-e-opencv/)")
		st.text("Felipe Meganha")
		st.success("Felipe Meganha @FMeganha")



if __name__ == '__main__':
		main()	