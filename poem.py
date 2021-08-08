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
paddedsequence = 
