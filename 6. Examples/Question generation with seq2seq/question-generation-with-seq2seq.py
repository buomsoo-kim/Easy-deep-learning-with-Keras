import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import *

url = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'

# import dataset using read_table() function in pandas
data = pd.read_table(url, sep ='\t')

# select question pairs that are "duplicate"
data = data[data['is_duplicate'] == 1]

# number of samples and minimum count of words to include
num_samples = 10000
min_count = 5

q1 = list(data['question1'])[:num_samples]
q2 = list(data['question2'])[:num_samples]

# create a list to put in all words (tokens) in input and target dataset
input_words = []
target_words = []

for i in range(len(q1)):
    for token in q1[i].split():
        input_words.append(token)
    for token in q2[i].split():
        target_words.append(token)

# convert lists into sets to choose only unique tokens
unique_input_words = set(input_words)
unique_target_words = set(target_words)

to_delete = []
for token in unique_input_words:
    if input_words.count(token) < min_count:
        to_delete.append(token)

for token in to_delete:
    unique_input_words.remove(token)

to_delete = []
for token in unique_target_words:
    if target_words.count(token) < min_count:
        to_delete.append(token)

for token in to_delete:
    unique_target_words.remove(token)

q1 = [q.split() for q in q1]
q2 = [('@ ' + q + ' #').split() for q in q2]

# also add symbols to unique token set
unique_target_words.update(['@', '#'])

for i in range(len(q1)):
    q1[i] = [token for token in q1[i] if token in unique_input_words]
    q2[i] = [token for token in q2[i] if token in unique_target_words]

del input_words
del target_words

unique_input_words = sorted(list(unique_input_words))
unique_target_words = sorted(list(unique_target_words))

# number of tokens => dimensionality of one-hot encoding space of tokens
num_encoder_tokens = len(unique_input_words)
num_decoder_tokens = len(unique_target_words)

# maximum sequence length 
max_encoder_seq_len = max([len(q) for q in q1])
max_decoder_seq_len = max([len(q) for q in q2])

print('Total Number of samples: ', len(q1))
print('Number of unique input tokens (words): ', num_encoder_tokens)
print('Number of unique output tokens (words): ', num_decoder_tokens)
print('Max seq length for inputs: ', max_encoder_seq_len)
print('Max seq length for outputs: ', max_decoder_seq_len)

input_token_idx = dict([(token, i) for i, token in enumerate(unique_input_words)])
target_token_idx = dict([(token, i) for i, token in enumerate(unique_target_words)])

encoder_input = np.zeros((len(q1), max_encoder_seq_len, num_encoder_tokens), dtype = 'float32')
decoder_input = np.zeros((len(q1), max_decoder_seq_len, num_decoder_tokens), dtype = 'float32')
decoder_target = np.zeros((len(q1), max_decoder_seq_len, num_decoder_tokens), dtype = 'float32')

for i, (x, y) in enumerate(zip(q1, q2)):
    for t, token in enumerate(x):
        encoder_input[i, t, input_token_idx[token]] = 1.
    for t, token in enumerate(y):
        decoder_input[i, t, target_token_idx[token]] = 1.
        if t > 0:
            decoder_target[i, t-1, target_token_idx[token]] = 1.

latent_dim = 300

encoder_inputs = Input(shape = (None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state = True)
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape = (None, num_decoder_tokens))
lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
decoder_outputs, _, _ = lstm(decoder_inputs, initial_state = encoder_states)
dense = Dense(num_decoder_tokens, activation = 'softmax')
decoder_outputs = dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.fit([encoder_input, decoder_input], decoder_target, batch_size = 100, epochs = 100, verbose = 0)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape = (latent_dim, ))
decoder_state_input_c = Input(shape = (latent_dim, ))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = lstm(decoder_inputs, initial_state = decoder_state_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

reverse_input_token_idx = dict((i, token) for token, i in input_token_idx.items())
reverse_target_token_idx = dict((i, token) for token, i in target_token_idx.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_idx['@']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_token_idx[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '#' or
           len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

for idx in range(100):
    input_seq = encoder_input[idx: idx+1]
    decoded_sent = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', q1[idx])
    print('Decoded sentence:', decoded_sent)