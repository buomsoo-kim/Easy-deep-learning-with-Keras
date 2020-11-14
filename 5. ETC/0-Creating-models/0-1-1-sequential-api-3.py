from IPython.display import SVG
from sklearn.datasets import load_digits
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, Activation

data = load_digits()

X_data = data.images
y_data = data.target

# flatten X_data
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1]*X_data.shape[2])

model = Sequential()
model.add(Dense(10, input_shape = X_data.shape[1:], activation = 'relu', name = 'Input_layer'))
model.add(Dense(50, activation = 'relu', name = 'First_hidden_layer'))
model.add(Dense(10, activation = 'softmax', name = 'Output_layer'))
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))