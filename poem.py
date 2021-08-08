import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokennizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

path_to_file = ""

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

tokenizer = Tokenizer()
tokenizer.fit_on_text(text)

sequences = tokenizer.text_to_sequences(sentences)

#padding sequence of text
max_sequence_len = max([len(x) for x in sequences])
paddedsequence = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))



def build_model(seqence_length, embedding_dim, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(seqence_length, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.Bidirectional(LSTM(150)),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

sequence_size = len(sequences)
embedding_dim = 256
