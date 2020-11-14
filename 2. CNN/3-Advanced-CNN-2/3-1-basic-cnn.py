import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Embedding, Flatten
from keras import optimizers

num_features = 3000
sequence_length = 300
embedding_dimension = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_features)

X_train = pad_sequences(X_train, maxlen = sequence_length)
X_test = pad_sequences(X_test, maxlen = sequence_length)

def imdb_cnn():
    model = Sequential()
    
    # use Embedding layer to create vector representation of each word => it is fine-tuned every iteration
    model.add(Embedding(input_dim = 3000, output_dim = embedding_dimension, input_length = sequence_length))
    model.add(Conv1D(filters = 50, kernel_size = 5, strides = 1, padding = 'valid'))
    model.add(MaxPooling1D(2, padding = 'valid'))
    
    model.add(Flatten())
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    adam = optimizers.Adam(lr = 0.001)
    
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    
    return model

model = imdb_cnn()

history = model.fit(X_train, y_train, batch_size = 50, epochs = 100, validation_split = 0.2, verbose = 0)

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.legend(['training', 'validation'], loc = 'upper left')
# plt.show()

results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])