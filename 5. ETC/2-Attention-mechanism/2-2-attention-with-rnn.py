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

y_data = to_categorical(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 777)

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def attention_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(10, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = attention_lstm()
model.fit([X_train], y_train, epochs = 100, batch_size = 64, validation_split = 0.3, verbose = 0)

attention_vector = get_activations(model, [X_train[0]], print_shape_only=True, layer_name = 'attention_vec')[0].reshape(8,8)

plt.imshow(attention_vector)
plt.show()