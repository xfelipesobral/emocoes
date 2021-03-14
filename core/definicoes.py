from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

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

# Define emoções bases
emocao = {
    0: 'Raiva',
    1: 'Nojo',
    2: 'Medo',
    3: 'Feliz',
    4: 'Triste',
    5: 'Surpreso',
    6: 'Neutro'
}
