import camera

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.utils import to_categorical

# carrega variáveis
emocao = camera.emocoesBase
modelo = camera.modelo

# importa base
data = pd.read_csv('modelo/fer2013.csv')
data.head()

# separa base de testes
def separar(dados):
  imgPixels = np.zeros(shape=(len(dados), 48, 48)) # Transforma em uma matriz, para separar cada pixel da imagem
  imgEmocao = np.array(list(map(int, dados['emotion']))) # Coloca emoções da imagem em um único array

  # Separa os pixels em linha para uma matriz 48x48 (Que será o tamanho da imagem final)
  for i, row in enumerate(dados.index):
      imagem = np.fromstring(dados.loc[row, 'pixels'], dtype=int, sep=' ')
      imagem = np.reshape(imagem, (48, 48))
      imgPixels[i] = imagem

  return imgPixels, imgEmocao

# Imprime gráfico das emoções
def imprimir(imgsTestePixels, resultado, i):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    txtEmocao = emocao.values()

    axs[0].imshow(imgsTestePixels[i], 'gray')

    axs[1].bar(txtEmocao, resultado[i], color='blue', alpha=0.7)
    axs[1].grid()

    plt.show()

# Imagens e emoções separadas por categoria
imgsTestePixels, emcTeste = separar(data[data['Usage']=='PublicTest'])
imgsTeste = imgsTestePixels.reshape((imgsTestePixels.shape[0], 48, 48, 1))
imgsTeste = imgsTeste.astype('float32')/255
emcTeste = to_categorical(emcTeste)

# Passa imagens para o modelo
resultado = modelo.predict(imgsTeste)

# Imprime resultados
imprimir(imgsTestePixels, resultado, 200)
imprimir(imgsTestePixels, resultado, 40)
imprimir(imgsTestePixels, resultado, 120)
imprimir(imgsTestePixels, resultado, 130)
imprimir(imgsTestePixels, resultado, 169)