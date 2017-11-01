import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Input, concatenate, multiply, Dense, Permute, Reshape, LSTM, Activation
from keras import optimizers
from keras.utils.np_utils import to_categorical
import keras.backend as K

data = datasets.load_digits()

X_data = data.images
y_data = data.target

X_data = X_data.reshape(X_data.shape[0], 64)
y_data = to_categorical(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 777)

def attention_mlp():
    input_layer = Input(shape = (X_train.shape[1],))
    attention_probs = Dense(X_train.shape[1], activation = 'softmax', name='attention_vec')(input_layer)    # attention vector is achieved here
    attention_mul = multiply([input_layer, attention_probs], name='attention_mul')
    
    attention_mul = Dense(50)(attention_mul)
    output_layer = Dense(10, activation = 'softmax')(attention_mul)
    
    model = Model(inputs = [input_layer], outputs = output_layer)
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = attention_mlp()
model.fit([X_train], y_train, epochs = 50, batch_size = 50, verbose = 0)

# function to obtain activation of each layer
def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

attention_vector = get_activations(model,[X_train[0]], print_shape_only =True)[0]

attention_vector = attention_vector.reshape(8,8)
plt.imshow(attention_vector)
plt.show()

X7 = []
for i in range(len(X_train)):
    if y_train[i][7] == 1:
        X7.append(X_train[i])
X7 = np.array(X7)

attention_vector = get_activations(model, X7, print_shape_only =True)[0]
attention_vector = np.mean(attention_vector, axis = 0).reshape(8, 8)

plt.imshow(attention_vector)
plt.show()