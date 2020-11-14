from IPython.display import SVG
from sklearn.datasets import load_digits
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential, Model
from keras.layers import *

data = load_digits()

X_data = data.images
y_data = data.target

# flatten X_data
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1]*X_data.shape[2])

# creating layers
input_layer = Input(shape = X_data.shape[1:])
activation_1 = Activation('relu')(input_layer)
hidden_layer = Dense(50)(activation_1)
activation_2 = Activation('relu')(hidden_layer)
output_layer = Dense(10, activation = 'softmax')(activation_2)

# creating model
model = Model(inputs = input_layer, outputs = output_layer)

print(model.summary())
