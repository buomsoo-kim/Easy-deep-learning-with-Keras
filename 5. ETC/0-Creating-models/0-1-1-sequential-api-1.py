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

model = Sequential([Dense(10, input_shape = X_data.shape[1:]), Dense(10, activation = 'softmax')])
SVG(model_to_dot(model).create(prog='dot', format='svg'))

