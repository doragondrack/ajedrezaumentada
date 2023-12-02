import cv2
import numpy as np
from calibracion import *

# Variables globales para el estado
aruco_seleccionado = None
posicion_seleccionada = None
cel = 0
#posicion inicial de la ficha
idd = 1
def piramide(esquinas, frame):
    # calcular las esquinas
    c1 = (esquinas[i][0][0][0], esquinas[i][0][0][1])
    c2 = (esquinas[i][0][1][0], esquinas[i][0][1][1])
    c3 = (esquinas[i][0][2][0], esquinas[i][0][2][1])
    c4 = (esquinas[i][0][3][0], esquinas[i][0][3][1])
    v1, v2 = c1[0], c1[1]
    v3, v4 = c2[0], c2[1]
    v5, v6 = c3[0], c3[1]
    v7, v8 = c4[0], c4[1]
    # Dibujar líneas que conectan los puntos de la base piramide
    bs1 = cv2.line(frame, (int(v1), int(v2)), (int(v3), int(v4)), (255,0,255),1)
    bs2 = cv2.line(frame, (int(v5), int(v6)), (int(v7), int(v8)), (255,0,255),1)
    bs3 = cv2.line(frame, (int(v1), int(v2)), (int(v7), int(v8)), (255,0,255),1)
    bs4 = cv2.line(frame, (int(v3), int(v4)), (int(v5), int(v6)), (255,0,255),1)
    cex1, cey1 = (v1 + v5) // 2, (v2+v6)//2
    cex2, cey2 = (v3 + v7) // 2, (v4+v8)//2
    # Dibujar líneas que conectan los puntos de la piramide
    p1 = cv2.line(frame, (int(v1), int(v2)), (int(cex1), int(cey1 - 100)), (255,0,255),1)
    p2 = cv2.line(frame, (int(v5), int(v6)), (int(cex1), int(cey1 - 100)), (255,0,255),1)
    p3 = cv2.line(frame, (int(v3), int(v4)), (int(cex1), int(cey2 - 100)), (255,0,255),1)
    p4 = cv2.line(frame, (int(v7), int(v8)), (int(cex1), int(cey2 - 100)), (255,0,255),1)

    # Dibujar el aruco seleccionado de manera diferente (por ejemplo, con un color diferente)
    if aruco_seleccionado is not None and aruco_seleccionado in ids:
        index = ids.tolist().index(aruco_seleccionado)
        c1, c2, c3, c4 = esquinas[index][0][:4]
        cv2.polylines(frame, [np.int32([c1, c2, c3, c4])], True, (0, 255, 0), 2)

def click_event(event, x, y, flags, param):
    global aruco_seleccionado, posicion_seleccionada
    global idd , cel

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(len(esquinas)):
            x_min = min(esquinas[i][0][:, 0])
            x_max = max(esquinas[i][0][:, 0])
            y_min = min(esquinas[i][0][:, 1])
            y_max = max(esquinas[i][0][:, 1])

            if x_min < x < x_max and y_min < y < y_max:
                if ids[i] == idd:
                    # Seleccionar aruco
                    aruco_seleccionado = ids[i]
                    posicion_seleccionada = (x, y)
                    cel = 1
                if ids[i] != idd and cel == 1:
                    # Mover aruco
                    idd = ids[i]
                    print(idd,ids[i])
                    aruco_seleccionado = None
                    posicion_seleccionada = None
                    cel = 0
                #print("¡Hola! Has tocado el ArUco con ID:", ids[i],idd,cel)

# Configurar la cámara de OpenCV
cap = cv2.VideoCapture(0)

# Configurar el diccionario ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Configurar el detector ArUco
aruco_params = cv2.aruco.DetectorParameters_create()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 5

# Crear el tablero de ajedrez
board = cv2.aruco.GridBoard_create(5, 7, 0.04, 0.01, aruco_dict)

#se calibra la camara
calibracion = calibracion()
cameraMatrix, distCoeffs = calibracion.calibracion_cam()

#se inicia la pantalla y el click
cv2.namedWindow('ArUco Detection with Cube')
cv2.setMouseCallback('ArUco Detection with Cube', click_event)
# Dentro del bucle principal
while True:
    # Leer un cuadro desde la cámara
    ret, frame = cap.read()

    # Asegurarse de que frame sea una matriz NumPy válida
    if not ret or frame is None:
        continue

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de marcadores ArUco
    esquinas, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(esquinas, ids, board, cameraMatrix, distCoeffs,None,None)
    if ids is not None and len(ids) > 0:
        # Dibujar los contornos y los IDs de los marcadores ArUco
        cv2.aruco.drawDetectedMarkers(frame, esquinas, ids)
        for i in range(len(ids)):
            # Verificar si el ArUco específico fue detectado
            if ids[i] == idd:#que aruco quiere que le dibuje ensima
                #dibujar ensima del aruco
                piramide(esquinas,frame)
    if retval:
            # Dibujar el eje de la tabla
            cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
    # Mostrar la imagen resultante
    cv2.imshow('ArUco Detection with Cube', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()