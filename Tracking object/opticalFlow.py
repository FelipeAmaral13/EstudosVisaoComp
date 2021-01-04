import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#Setar parametros para ShiTomasi 
feature_params = dict(maxCornes = 100, qualityLevel=0.3, minDistance=7, blockSize=7)

#Setar params do Lucas-kanade 
lucas_kanade_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Pegar o primeiro frame e localizar cantos
ret, prev_frame = cap.read()
prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

#Primeiro canto
prev_corners = cv2.goodFeaturesToTrack(prev_gray , mask=None, **features_params)

#Criando uma mascara da imagem para desenhar
mask = np.zeros_like(frame)

while True:
	ret, frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Calculo do fluxo optico
	new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_corners, None, **lucas_kanade_params)

	# Seelcionar e armazenar os pontos bons
	good_new = new_corners[status==1]
	good_old = prev_corners[status==1]

	#Desenhar o rastreio
	for i,(new,old) in enumerate(zip(good_new, good_old)):
		a, b = new.ravel()
		c, d = old.ravel()
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		frame = cv2.circle(frame, (a,b), 5, color[i].tolist(),-1)

	img = cv2.add(frame,mask)

	#Mostrar o fluxo Optico
	cv2.imshow('Optical Flow- Lucas-Kanade',img)
	k = cv2.waitKey(30) & 0xff
  	if k == 27:
		  break
	prev_gray = frame_gray.copy()
	prev_corners = good_new.reshape(-1,1,2)
	

camera.release()
cv2.destroyAllWindows()
