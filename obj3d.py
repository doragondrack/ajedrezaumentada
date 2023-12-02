import cv2
import numpy as np
from objloader_simple import *
from calibracion import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective, gluLookAt
import glfw
import glad

# Función para inicializar OpenGL
def init_opengl(width, height):
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_NORMALIZE)
    glClearColor(0.0, 0.0, 0.0, 0.0)

# Función para renderizar el objeto 3D
def draw_obj(obj, rvec, tvec):
    glPushMatrix()

    # Extract the rotation and translation components
    rvec = rvec.squeeze()
    tvec = tvec.squeeze()

    glTranslatef(tvec[0], tvec[1], tvec[2])
    glRotatef(np.degrees(rvec[0]), 1, 0, 0)
    glRotatef(np.degrees(rvec[1]), 0, 1, 0)
    glRotatef(np.degrees(rvec[2]), 0, 0, 1)
    glBegin(GL_TRIANGLES)
    for face in obj.faces:
        for i in range(3):
            vertex_index = face[0][i] - 1
            normal_index = face[1][i] - 1
            glVertex3fv(obj.vertices[vertex_index])
            glNormal3fv(obj.normals[normal_index])
    glEnd()
    glPopMatrix()

# Cargar el modelo 3D
obj = OBJ("model.obj", swapyz=True)

# Configurar el detector de ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters_create()
cameraMatrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((4, 1), dtype=np.float32)
# Inicializar la cámara
cap = cv2.VideoCapture(0)
#calibracion = calibracion()
#cameraMatrix, distCoeffs = calibracion.calibracion_cam()
# Inicializar GLFW y crear una ventana
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
width, height = 640, 480
window = glfw.create_window(width, height, "AR Overlay", None, None)
glfw.make_context_current(window)

# Inicializar glad

# Inicializar OpenGL
init_opengl(width, height)
while  not glfw.window_should_close(window):
    glfw.poll_events()
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los marcadores ArUco
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            # Dibujar el contorno alrededor del ArUco
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Obtener la matriz de transformación
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, cameraMatrix, distCoeffs)
            print(rvec,tvec)
            # Proyectar el modelo 3D sobre el ArUco
            draw_obj(obj, rvec, tvec)

    # Mostrar el resultado
    glfw.swap_buffers(window)
    cv2.imshow("AR Overlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
glfw.terminate()