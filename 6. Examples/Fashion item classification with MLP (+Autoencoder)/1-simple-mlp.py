import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras import optimizers

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def simple_mlp():
    inputs = Input(shape = (X_train.shape[1],))
    l = Dense(100, kernel_initializer = 'he_normal')(inputs)
    l = Activation('selu')(l)
    l = BatchNormalization()(l)
    l = Dense(100, kernel_initializer = 'he_normal')(l)
    l = Activation('selu')(l)
    l = BatchNormalization()(l)
    outputs = Dense(10, activation = 'softmax')(l)
    
    adam = optimizers.adam(lr = 0.001)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = simple_mlp()

history = model.fit(X_train, y_train, epochs = 100, batch_size = 200, validation_split = 0.3, verbose = 0)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])