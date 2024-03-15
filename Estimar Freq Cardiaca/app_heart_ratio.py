import sys
import cv2
import numpy as np
from scipy.signal import find_peaks
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QGridLayout, QHBoxLayout

class WebcamCaptureApp(QMainWindow):
    def __init__(self):
        """Inicializador da classe."""
        super().__init__()

        self.setWindowTitle("Webcam Heart Ratio")
        self.setGeometry(100, 100, 640, 520)

        # Cria botões
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        # Conecta os sinais dos botões aos slots
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)

        # Cria widget de exibição de vídeo
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        # Cria rótulo para BPM
        self.bpm_label = QLabel("BPM: 0")
        self.bpm_label.setAlignment(Qt.AlignCenter)

        # Cria layout para botões
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Cria layout para BPM
        bpm_layout = QVBoxLayout()
        bpm_layout.addWidget(self.bpm_label)

        # Cria layout para conter o layout dos botões e o layout do BPM
        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(bpm_layout)
        bottom_layout.addLayout(button_layout)

        # Cria um layout de grade
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.video_label, 0, 0, 1, 2)  # Rótulo de vídeo abrangendo duas colunas
        grid_layout.addLayout(bottom_layout, 1, 0, 1, 2)  # Layout inferior abrangendo duas colunas

        central_widget = QWidget()
        central_widget.setLayout(grid_layout)
        self.setCentralWidget(central_widget)

        # Captura de vídeo
        self.capture = cv2.VideoCapture(1)  # Usa a webcam padrão (mude se necessário)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.recording_flag = False
        self.output = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.bpm = 0
        self.min_peak_height = 10
        self.min_peak_distance = 30

    def start_capture(self):
        """Inicia a captura de vídeo."""
        if not self.recording_flag:
            self.output = cv2.VideoWriter('captured_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
            self.recording_flag = True
            self.timer.start(30)  # Atualiza o quadro a cada 30 milissegundos

    def stop_capture(self):
        """Interrompe a captura de vídeo."""
        if self.recording_flag:
            self.output.release()
            self.recording_flag = False
            self.timer.stop()

    def update_frame(self):
        """Atualiza o quadro do vídeo."""
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detecte faces na imagem
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            bpm_sum = 0
            bpm_count = 0
            
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
                peaks, _ = find_peaks(filtered.ravel(), height=self.min_peak_height, distance=self.min_peak_distance)

                # Contagem dos picos (batimentos cardíacos)
                heart_rate = len(peaks)
                bpm_sum += heart_rate
                bpm_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(frame, f"BPM: {heart_rate} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Atualizar o BPM
            if bpm_count > 0:
                self.bpm = bpm_sum / bpm_count
                self.bpm_label.setText(f"BPM: {self.bpm:.2f}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)


    def closeEvent(self, event):
        self.capture.release()
        if self.output:
            self.output.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamCaptureApp()
    window.show()
    sys.exit(app.exec())
