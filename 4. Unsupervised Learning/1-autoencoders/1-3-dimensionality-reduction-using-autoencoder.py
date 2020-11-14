import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.layers import Input, Dense
from keras.models import Model

data = datasets.load_digits()

X_data = data.images
y_data = data.target

X_data = X_data.reshape(X_data.shape[0], 64)

# fit in data instances into interval [0,1]
X_data = X_data / 16.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 777)

# define coding dimension. Coding dimension will be the size of reduced data dimension
code_dim = 16

def auto_encoder_model():    
    inputs = Input(shape = (X_train.shape[1],), name = 'input')                         # input layer
    code = Dense(code_dim, activation = 'relu', name = 'code')(inputs)                  # hidden layer => represents "codes"
    outputs = Dense(X_train.shape[1], activation = 'softmax', name = 'output')(code)    # output layer

    auto_encoder = Model(inputs = inputs, outputs = outputs)

    encoder = Model(inputs = inputs, outputs = code)

    decoder_input = Input(shape = (code_dim,))
    decoder_output = auto_encoder.layers[-1]
    decoder = Model(inputs = decoder_input, outputs = decoder_output(decoder_input))

    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto_encoder

encoder, decoder, auto_encoder = auto_encoder_model()
auto_encoder.fit(X_train, X_train, epochs = 100, batch_size = 50, validation_data = (X_test, X_test), verbose = 0)

# generate reduced data by using "encoders"
training_data_reduced = encoder.predict(X_train)
test_data_reduced = encoder.predict(X_test)

print(training_data_reduced.shape)
print(test_data_reduced.shape)

print(training_data_reduced[0])    # first insance of reduced training data
print(test_data_reduced[0])        # first instance of reduced test data