from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.optimizers import Adam

# Cria modelo
modelo = Sequential()
modelo.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu',  input_shape=(48, 48,1)))
modelo.add(AveragePooling2D())
modelo.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
modelo.add(AveragePooling2D())
modelo.add(Flatten())
modelo.add(Dense(units=120, activation='relu'))
modelo.add(Dense(units=84, activation='relu'))
modelo.add(Dense(units=7, activation = 'softmax'))

modelo.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy',  metrics=['accuracy']) # Define otimizador
modelo.summary() # Imprime modelo montado

modelo.load_weights('modelo/modelo.hdf5') # Carrega modelo treinado

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
