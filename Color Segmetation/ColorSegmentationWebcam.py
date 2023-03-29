import cv2
import numpy as np

class CameraCapture:
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
    def range_HSV(self, frame, H: int, S: int, V: int):
        red_low = np.array([H, S, V])
        red_high = np.array([255, 255, 255])
        red_mask = cv2.inRange(frame, red_low, red_high)
        red = cv2.bitwise_and(frame, frame, mask=red_mask)
        return red

    def start_capture(self):
        while True:
            ret, frame = self.cap.read()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_hsv = self.range_HSV(hsv_frame, 160, 115, 85)
            cv2.imshow("RED", color_hsv)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == '__main__':
    cap = CameraCapture()
    cap.start_capture()
