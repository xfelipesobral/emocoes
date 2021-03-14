import cv2
import numpy as np 
import matplotlib.pyplot as plt

from core.definicoes import *
def imprimir(base, resultado, i):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    txtEmocao = emocao.values()

    axs[0].imshow(base[i], 'gray')

    axs[1].bar(txtEmocao, resultado[i], color='blue', alpha=0.7)
    axs[1].grid()

    plt.show()

def toMatriz(linha):
    arr = np.zeros(shape=(1, 48, 48))
    imagem = np.fromstring(linha, dtype=int, sep=' ')
    imagem = np.reshape(imagem, (48, 48))
    arr[0] = imagem

    return arr

def processaImagem(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    strPixels = ''
    for i in cinza:
        for b in i:
            strPixels += str(b)+' '

    matriz = toMatriz(strPixels)

    # Formata
    img = matriz.reshape((matriz.shape[0], 48, 48, 1))
    img = img.astype('float32')/255

    resultado = modelo.predict(img)

    imprimir(matriz, resultado, 0)
    #cv2.imshow('Imagem', cinza)