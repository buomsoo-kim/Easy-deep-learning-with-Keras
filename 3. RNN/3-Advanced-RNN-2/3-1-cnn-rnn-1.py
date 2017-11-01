import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Input, Activation, Reshape, concatenate
from keras import optimizers

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Conv2D(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Reshape(target_shape = (16*16, 50)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(10))
model.add(Activation('softmax'))
adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, verbose = 1)

results = model.evaluate(X_test, y_test)
print('Test Accuracy: ', results[1])