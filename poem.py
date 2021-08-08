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

BATCH_SIZE = 64

#Create Bidirectional RNN
def create_model(seqence_length, embedding_dim, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(seqence_length, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.Bidirectional(LSTM(128)),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

sequence_size = len(sequences)
embedding_dim = 256


model = create_model(sequence_size, embedding_dim, BATCH_SIZE)
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# create predictors and label
total_words = len(tokenizer.word_index) + 1
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


history = model.fit(xs, ys, epochs=100, verbose=1)

def generate_text():
    seed_text = "I've got a bad feeling about this"
    next_words = 100
  
    for _ in range(next_words):
	    token_list = tokenizer.texts_to_sequences([seed_text])[0]
	    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	    predicted = model.predict_classes(token_list, verbose=0)
	    output_word = ""
	    for word, index in tokenizer.word_index.items():
		    if index == predicted:
			    output_word = word
			    break
	    seed_text += " " + output_word
    print(seed_text)
