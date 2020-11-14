from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras.preprocessing.sequence import pad_sequences

sentences = ['Curiosity killed the cat.', 'But satisfaction brought it back']

tk = Tokenizer()    # create Tokenizer instance
tk.fit_on_texts(sentences)    # tokenizer should be fit with text data in advance

# text to sequences
seq = tk.texts_to_sequences(sentences)
print(seq)

# one-hot encoding of sentences
mat = tk.sequences_to_matrix(seq)
print(mat)

## zero-padding sequences
# if set padding to 'pre', zeros are appended to start of sentences
pad_seq = pad_sequences(seq, padding='pre')     
print(pad_seq)

# if set padding to 'post', zeros are appended to end of sentences
pad_seq = pad_sequences(seq, padding='post')
print(pad_seq)