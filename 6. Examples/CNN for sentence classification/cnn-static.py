import os
import numpy as np
import re
import random

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from nltk.corpus import stopwords

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, concatenate, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm

stop_words = set(stopwords.words('english'))    # english stopwords from nltk

# lists to contain reviews and labels (pos/neg)
review_list = []
labels_list = []

num_instances = 5000   # number of instances to consider. Actual number of whole data instances would be (num_instances * 4)

# reading text files. 
files = os.listdir('aclImdb/train/pos')[:num_instances]
for file in files:
    with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        sentence = [word for word in f.read().split() if word not in stop_words]
        sentence = [word.lower() for word in sentence if re.match('^[a-zA-Z]+', word)]
        f.close()
    review_list.append(sentence)
    labels_list.append(1)

files = os.listdir('aclImdb/train/neg')[:num_instances]
for file in files:
    review = ''
    with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        sentence = [word for word in f.read().split() if word not in stop_words]
        sentence = [word.lower() for word in sentence if re.match('^[a-zA-Z]+', word)]
        f.close()
    review_list.append(sentence)
    labels_list.append(0)

files = os.listdir('aclImdb/test/pos')[:num_instances]
for file in files:
    review = ''
    with open('aclImdb/test/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        sentence = [word for word in f.read().split() if word not in stop_words]
        sentence = [word.lower() for word in sentence if re.match('^[a-zA-Z]+', word)]
        f.close()
    review_list.append(sentence)
    labels_list.append(1)

files = os.listdir('aclImdb/test/neg')[:num_instances]
for file in files:
    review = ''
    with open('aclImdb/test/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        sentence = [word for word in f.read().split() if word not in stop_words]
        sentence = [word.lower() for word in sentence if re.match('^[a-zA-Z]+', word)]
        f.close()
    review_list.append(sentence)
    labels_list.append(0)

threshold = max_len = 500   # in order to cut out excessively long sentences, define a thresold (i.e., a maximum number of words to be considered)

# if the length of the sentence is longer than the threshold, exclude that sentence
for i in range(len(review_list)):
    if len(review_list[i]) > threshold :
        review_list[i] = None
        labels_list[i] = None

review_list = [rev for rev in review_list if rev is not None] 
labels_list = [rev for rev in labels_list if rev is not None] 

embed_dim = 100    # assign the dimension of the embedding space

model = Word2Vec(sentences = review_list, size = embed_dim, sg = 1, window = 5, min_count = 1, iter = 10, workers = Pool()._processes)
model.init_sims(replace = True)    

# create a 3-D numpy array to carry X data
X_data = np.zeros((len(review_list), max_len, embed_dim))

for i in range(len(review_list)):
    for j in range(max_len):
        try:
            X_data[i][j] = model[review_list[i][j]]
        except:
            pass   # if the word is not included in the embedding space, assign zero (i.e., zero padding)

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], X_data.shape[2], 1)    # reshape the data into 4-D shape
print(X_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, labels_list, test_size = 0.3, random_state = 777)

# assign the hyperparameters of the model
filter_sizes = [3, 4, 5]
dropout_rate = 0.5
l2_constraint = 3.0

def convolution():
    inn = Input(shape = (max_len, embed_dim, 1))
    convolutions = []
    # we conduct three convolutions & poolings then concatenate them.
    for fs in filter_sizes:
        conv = Conv2D(filters = 100, kernel_size = (fs, embed_dim), strides = 1, padding = "valid")(inn)
        nonlinearity = Activation('relu')(conv)
        maxpool = MaxPooling2D(pool_size = (max_len - fs + 1, 1), padding = "valid")(nonlinearity)
        convolutions.append(maxpool)
    outt = concatenate(convolutions)
    model = Model(input = inn, output = outt)
    
    return model

def cnn_model():
    convolutions = convolution()
    
    model = Sequential()
    model.add(convolutions)
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(1, kernel_constraint=maxnorm(l2_constraint), activation = 'sigmoid'))
    
    adam = optimizers.Adam(lr = 0.01)
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

cnn = cnn_model()
cnn.summary()    # summary of the model

cnn_model = KerasClassifier(build_fn = cnn_model, epochs = 100, batch_size = 50, verbose = 1)
cnn_model.fit(X_train, y_train)

y_pred = cnn_model.predict(X_test)
print(accuracy_score(y_test, y_pred))