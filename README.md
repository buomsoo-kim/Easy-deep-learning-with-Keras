# Easy-deep-learning-with-Keras

If you are unfamiliar with data preprocessing, first review **NumPy & Pandas** sections of ```Python for data analysis``` repository materials (https://github.com/buomsoo-kim/Python-for-data-analysis)

Materials in this repository are for educational purposes.
Source code is written in Python 3.6+ & Keras ver 2.0+ (Using TensorFlow backend - For advanced topics, basic understanding of TensorFlow mechanics is necessary)

## 1. Multilayer Perceptrons

### 1) Basics of MLP
- Regression tasks with MLP
- Classification tasks with MLP

### 2) Advanced MLP - 1
- Weight initialization schemes
- Activation functions (Nonlinearity)
- Batch normalization
- Optimizers
- Dropout
- Ensemble of models

### 3) Advanced MLP - 2
- Putting it altogether

## 2. Convolutional Neural Networks

### 1) Basic CNN
- Basics of CNN architecture

### 2) Advanced CNN - 1
- Getting deeper with CNNs

### 3) Advanced CNN - 2
- CNN for sentence classification (imdb)

### 4) Using pretrained models
- Importing models already trained on ImageNet dataset (keras.applications)

## 3. Recurrent Neural Networks

### 1) Basic RNN
- Understanding RNN architecture
- Vanilla RNN (SimpleRNN)
- Stacked vanilla RNN
- LSTM
- Stacked LSTM

### 2) Advanced RNN - 1
- Deep RNNs
- Bidirectional RNNs
- Deep bidirectional RNNs

### 3) Advanced RNN - 2
- CNN-RNN

### 4) Advanced RNN - 3
- CuDNN LSTM
- CuDNN GRU

## 4. Unsupervised Learning

### 1) Autoencoders
- Autoencoder basics
- Convolutional autoencoder
- Dimensionality reduction using autoencoder

## 5. ETC

### 0) Creating models
- Sequential API
- Model Functional API

### 1) Image processing
 - Importing images

### 2) Keras callbacks
 - ModelCheckpoint
 - EarlyStopping
 - ReduceLROnPlateau
 
### 3) Using GPUs
 - Make your training process faster with CUDA & CuDNN
 
### 4) Model selection
 - Cross validation
 - Grid search
 - Random search

## 6. Examples

### 1) Digit Recognition with RNN
  - Simple RNN model
  - Stacked RNN model
  - Bidirectional RNN model
  - Simple LSTM model
  - Stacked LSTM model
  - Bidirectional LSTM model
  - Simple GRU model
  - Stacked GRU model
  - Bidirectional GRU model

### 2) Fashion item classification with MLP
  - Simple MLP
  - Autoencoder + MLP (dimensionality reduction)
  
### 3) Question generation with seq2seq (using Quora dataset)
  - Generating similar questions with seq2seq model

### 4) CNN for sentence classification
  - CNN-static implementation of Kim 2014 paper

## 7. Text Analytics
Section with emphasis on text data analytics

 ### 1) Text processing
 
 ### 2) Word embedding
 
 ### 3) CNNs for text data
   - 1-D Convolution for text analysis
   - CNN for setnence classification (Kim 2014)
   - Dynamic CNN for sentence modeling (Kalchbrenner et al 2014)
   - CNN for text categorization (Johnson and Zhang 2014)
