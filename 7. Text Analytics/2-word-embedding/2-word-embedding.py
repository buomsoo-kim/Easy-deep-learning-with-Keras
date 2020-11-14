from keras.models import Sequential
from keras.layers import *
from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils import to_categorical

# parameters to import dataset
num_words = 3000
maxlen = 50

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen)

X_train = sequence.pad_sequences(X_train, maxlen = maxlen, padding = 'post')
X_test = sequence.pad_sequences(X_test, maxlen = maxlen, padding = 'post')
y_train = to_categorical(y_train, num_classes = 46)
y_test = to_categorical(y_test, num_classes = 46)

input_dim = num_words
output_dim = 100     # we set dimensionality of embedding space as 100
input_length = maxlen

def reuters_model():
    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length))
    model.add(CuDNNGRU(50, return_sequences = False))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(46, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = reuters_model()

model.fit(X_train, y_train, epochs = 100, batch_size = 100, verbose = 0)
result = model.evaluate(X_test, y_test)
print('Test Accuracy: ', result[1])