import cv2
import numpy as np
import imutils
from imutils import perspective, contours


cap = cv2.VideoCapture(0)


while True:

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Filtro Gaussiano
    blurimg = cv2.GaussianBlur(gray, (5,5), 0)

    #Deteccao de bordas por Canny
    edges = cv2.Canny(blurimg, 100, 255)

    #Opercao Morfologica Fechamento
    img_dilate = cv2.dilate(edges, None, iterations=1)
    img_erode = cv2.erode(img_dilate, None, iterations=1)

    #kernel = np.ones((5,5),np.uint8) 
    #img_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) 

    #Contornos
    cnts = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, hir) = contours.sort_contours(cnts)

    #Remover contornos nos quais nao sao suficientemente grandes
    cnts = [x for in cnts if cv2.contourArea(x) > 100]

    cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
    print(len(cnts))

    #Gabarito (2cmx2cm)
    ref_obj = cnts[0]
    box = cv2.minAreaRect(ref_obj) #Calcula e retorna um retangulo delimitado de uma area min
    box = cv2.boxPoints(box)
    box = np.array(box, dtype='int')
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm

    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)

        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))

        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(tr, br)/pixel_per_cm
        cv2.putText(frame, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        



    cv2.imshow("Frame", frame)
    #cv2.imshow("Edges", edges)
    #cv2.imshow("Close", img_erode)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

