    
# Este script é usado para demonstrar a rede MobileNet-SSD usando o módulo de aprendizado profundo OpenCV.
#
# Ele funciona com o modelo retirado de https://github.com/chuanqi305/MobileNet-SSD/ que
# foi treinado na estrutura Caffe-SSD, https://github.com/weiliu89/caffe/tree/ssd.
# O modelo detecta objetos de 20 classes.
#
# Também o modelo TensorFlow do zoo de modelo de detecção de objeto TensorFlow pode ser usado para
# detectar objetos de 90 classes:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# A definição do gráfico de texto deve ser obtida de opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt

import numpy as np
import argparse

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Não é possível encontrar o módulo OpenCV Python. Se você o criou a partir de fontes sem instalação, '
                       'configure a variável de ambiente PYTHONPATH para o diretório "opencv_build_dir / lib" (com o subdiretório "python3", se necessário)')
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scriptpara rodar MobileNet-SSD object detection network '
                    'treinado com Caffe ou TensorFlow frameworks.')
    parser.add_argument("--video", help="Caminho do video. Se vazio, web cam será usada")
    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                      help='Caminho do arquivo text network: '
                                           'MobileNetSSD_deploy.prototxt do Caffe model ou '
                                           'ssd_mobilenet_v1_coco.pbtxt do opencv_extra do TensorFlow model')
    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                     help='Caminho dos pesos: '
                                          'MobileNetSSD_deploy.caffemodel do Caffe model ou '
                                          'frozen_inference_graph.pb do TensorFlow.')
    parser.add_argument("--num_classes", default=20, type=int,
                        help="Numeros de classes. É 20 do moledo do Caffe "
                             "https://github.com/chuanqi305/MobileNet-SSD/ e 90 do modelo "
                             "TensorFlow  do http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz")
    parser.add_argument("--thr", default=0.2, type=float, help="limiar de confiança para filtrar detecções fracas")
    args = parser.parse_args()

    if args.num_classes == 20:
        net = cv.dnn.readNetFromCaffe(args.prototxt, args.weights)
        swapRB = False
        classNames = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
    else:
        assert(args.num_classes == 90)
        net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
        swapRB = True
        classNames = { 0: 'background',
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }

    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    while True:
        # Frame Captura
        ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    


                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Deteccao", frame)
        if classNames[class_id] == 'person':
            cv.imwrite("SavarDeteccao.jpg", frame)
        if cv.waitKey(1) >= 0:
            break
