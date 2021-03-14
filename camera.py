import cv2
import face_recognition
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

camera = cv2.VideoCapture(0) # Inicializa câmera
faces = [] # Irá guardar a posição das faces na imagem

# Cria modelo
modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
modelo.add(MaxPool2D((2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPool2D((2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(Flatten())
modelo.add(Dense(64, activation='relu')) 
modelo.add(Dense(7, activation='softmax')) 

modelo.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',  metrics=['accuracy']) # Define otimizador
modelo.summary() # Imprime modelo montado

modelo.load_weights('modelo/treinado.hdf5') # Carrega modelo treinado

emocoesBase = {
    0: 'Raiva',
    1: 'Nojo',
    2: 'Medo',
    3: 'Feliz',
    4: 'Triste',
    5: 'Surpreso',
    6: 'Neutro'
}

########################################### 
# TESTE
def teste():
    print('teste')


#############################################



# Looping para ficar capturando
# while True:
#     ret, frame = camera.read()

#     redimensiona = cv2.resize(frame,  (0, 0), fx=0.25, fy=0.25) # tira 1/4
#     rgbFrame = redimensiona[:, :, ::-1]

#     faces = face_recognition.face_locations(rgbFrame)

#     for (top, right, bottom, left) in faces:
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         # Posicao da imagem
#         x = left - 40
#         y = top - 40
#         xy = right + 40
#         yx = bottom + 30
#         face = frame[y:yx, x:xy]
#         cv2.imwrite('teste.jpg', face)

#         # Desenha retângulo
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 1)

#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()