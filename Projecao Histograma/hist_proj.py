import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(self.image_path)

    def process_image(self):
        # convert to grayscale
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

        # smooth the image to avoid noises
        gray = cv2.medianBlur(gray,5)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # apply some dilation and erosion to join the gaps
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel ,iterations = 2)
        thresh = cv2.erode(thresh, kernel, iterations = 2)

        # Find the contours
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, find the bounding rectangle and draw it
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if h > 10:    
                cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h, x:x+w]

                height, width = roi.shape

                vertical_px = np.sum(roi, axis=0)
                normalize = vertical_px/255
                blankImage = np.zeros_like(roi)

                for idx, value in enumerate(normalize):
                    cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)

                img_concate = cv2.hconcat(
                    [self.img[y:y+h, x:x+w],  cv2.cvtColor(blankImage, cv2.COLOR_BGR2RGB)])

                plt.imshow(img_concate)
                plt.show()

image_processor = ImageProcessor('sof.png')
image_processor.process_image()
