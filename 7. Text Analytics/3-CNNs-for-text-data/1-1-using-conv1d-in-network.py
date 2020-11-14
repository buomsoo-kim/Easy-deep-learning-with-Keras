from keras.models import *
from keras.layers import *
from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# parameters to import dataset
def get_reuters(num_words = 3000, maxlen = 50):    
    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen)

    X_train = sequence.pad_sequences(X_train, maxlen = maxlen, padding = 'post')
    X_test = sequence.pad_sequences(X_test, maxlen = maxlen, padding = 'post')
    y_train = to_categorical(y_train, num_classes = 46)
    y_test = to_categorical(y_test, num_classes = 46)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    return X_train, X_test, y_train, y_test

num_words = 3000
max_len = 50
embed_size = 100

X_train, X_test, y_train, y_test = get_reuters(num_words, max_len)

def one_dim_convolution_model(num_words, embed_size, input_length):
    model = Sequential()
    model.add(Embedding(num_words, embed_size, input_length = max_len))
    model.add(Conv1D(50, 10, activation = 'relu'))
    model.add(MaxPooling1D(10))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(46, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
    return model
    
model = one_dim_convolution_model(num_words, embed_size, max_len)

history = model.fit(X_train, y_train, epochs = 100, batch_size = 100)
result = model.evaluate(X_test, y_test)
print('Test Accuracy: ', result[1])