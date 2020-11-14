from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

model = Sequential()

# Keras model with two hidden layer with 10 neurons each 
model.add(Dense(10, input_shape = (13,)))    # Input layer => input_shape should be explicitly designated
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(1))                          # Output layer => output dimension = 1 since it is regression problem

'''
This is equivalent to the above code block

>> model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
>> model.add(Dense(10, activation = 'sigmoid'))
>> model.add(Dense(10, activation = 'sigmoid'))
>> model.add(Dense(1))
'''

sgd = optimizers.SGD(lr = 0.01)    # stochastic gradient descent optimizer

model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])    # for regression problems, mean squared error (MSE) is often employed
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('mse: ', results[1])