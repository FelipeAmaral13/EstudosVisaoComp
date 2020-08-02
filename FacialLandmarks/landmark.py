from imutils import face_utils
import dlib
import cv2
 
#Repositorio das faces
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # Detectando as faces
    rects = detector(gray, 0)
    
    # Encontrando os keypoints.
    for (i, rect) in enumerate(rects):
        # Predicao e array do numpy.
        shape = predictor(gray, rect)
        for j in range(1,68):
            cv2.putText(frame, str(j), (shape.part(j).x, shape.part(j).y), 
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
            fontScale=0.3, 
            color=(0,0,255))
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
