import cv2
import numpy as np
from scipy.signal import find_peaks
import time

# Inicialize a webcam
cap = cv2.VideoCapture(1)

# Carregue o classificador de detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parâmetros para detecção de picos
MIN_PEAK_HEIGHT = 10
MIN_PEAK_DISTANCE = 30

# Variáveis para controle de tempo
start_time = time.time()
update_interval = 2  # Atualizar a cada segundo

# Variáveis para BPM
bpm = 0

while True:
    # Capture o quadro da webcam
    ret, frame = cap.read()

    # Converta o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte faces na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraia a região da testa (acima dos olhos)
        forehead_roi = gray[y:y+int(h/3), x:x+w]

        # Aplique um filtro passa-banda adaptativo
        filtered = cv2.bilateralFilter(forehead_roi, d=15, sigmaColor=75, sigmaSpace=75)

        # Transformada de Fourier para filtragem
        fft = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(fft))
        # Ajuste a frequência de corte com base na frequência cardíaca esperada
        fft[np.abs(freqs) > 0.1] = 0
        filtered = np.fft.ifft(fft).real

        # Detecte os picos no sinal filtrado
        peaks, _ = find_peaks(filtered.ravel(), height=MIN_PEAK_HEIGHT, distance=MIN_PEAK_DISTANCE)

        # Contagem dos picos (batimentos cardíacos)
        heart_rate = len(peaks)

        # Atualizar o BPM a cada segundo
        elapsed_time = time.time() - start_time
        if elapsed_time >= update_interval:
            bpm = heart_rate
            start_time = time.time()

        # Desenhe um retângulo na face detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Exiba o resultado na tela
        cv2.putText(frame, f"BPM: {bpm} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Exiba o sinal filtrado
        # cv2.imshow("Filtrado", filtered)

    # Exiba o quadro com as faces detectadas
    cv2.imshow("PPG", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a webcam e feche a janela
cap.release()
cv2.destroyAllWindows()
