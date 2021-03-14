#from core.definicoes import *
from core.definicoes import *

import cv2
import face_recognition

camera = cv2.VideoCapture(0) # Inicializa câmera
faces = [] # Irá guardar a posição das faces na imagem

# Looping para ficar capturando
while True:
    ret, frame = camera.read()

    redimensiona = cv2.resize(frame,  (0, 0), fx=0.25, fy=0.25) # tira 1/4
    rgbFrame = redimensiona[:, :, ::-1]

    faces = face_recognition.face_locations(rgbFrame)

    for (top, right, bottom, left) in faces:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Posicao da imagem
        x = left - 40
        y = top - 40
        xy = right + 40
        yx = bottom + 30
        face = frame[y:yx, x:xy]
        cv2.imwrite('teste.jpg', face)

        # Desenha retângulo
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()