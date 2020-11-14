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

model = create_model()
callbacks = [ModelCheckpoint(filepath = 'saved_model.hdf5', monitor='val_acc', verbose=1, mode='max')]
model.fit(X_train, y_train, epochs = 10, batch_size = 500, callbacks = callbacks, validation_data = (X_test, y_test))

results = model.evaluate(X_test, y_test)
print('Accuracy: ', results[1])


### Loading saved weights
another_model = create_model()
another_model.load_weights('saved_model.hdf5')
another_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

results = another_model.evaluate(X_test, y_test)
print('Accuracy: ', results[1])

### Selecting best model
callbacks = [ModelCheckpoint(filepath = 'best_model.hdf5', monitor='val_acc', verbose=1, save_best_only = True, mode='max')]
model = create_model()
model.fit(X_train, y_train, epochs = 10, batch_size = 500, callbacks = callbacks, validation_data = (X_test, y_test))

best_model = create_model()
best_model.load_weights('best_model.hdf5')
best_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

results = best_model.evaluate(X_test, y_test)
print('Accuracy: ', results[1])