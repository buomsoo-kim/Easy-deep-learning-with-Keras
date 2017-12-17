from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import *
from keras.layers import *

data = load_digits()

X_data = data.images
y_data = data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 777)

# reshaping X data => flatten into 1-dimensional
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def create_model():
    model = Sequential()
    model.add(Dense(100, input_shape = (X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# halve learning rate when validation loss has not reduced for more than 5 epochs
callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5)]

model = create_model()
model.fit(X_train, y_train, epochs = 20, batch_size = 500, callbacks = callbacks, validation_d

	results = model.evaluate(X_test, y_test)
	print('Accuracy: ', results[1])