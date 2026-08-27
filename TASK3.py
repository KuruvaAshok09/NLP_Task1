import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import spacy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("punkt_tab")
spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

corpus = """one disadvantage of using Best Of samping is that it may lead to
limited Exploration of the Models knowledge of Creativity. However, its
essential to carefully balance exploration based on the specific requirements
of the task or application"""

tokens = word_tokenize(corpus)

lemmatized_tokens = [token.lemma_ for token in nlp(corpus)]

all_tokens = tokens + lemmatized_tokens

text = " ".join(all_tokens)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

token_list = tokenizer.texts_to_sequences([text])[0]

input_sequences = []

for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i + 1]
    input_sequences.append(n_gram_sequence)

max_sequence_length = max(len(x) for x in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_length,
        padding="pre"
    )
)

x = input_sequences[:, :-1]
y = input_sequences[:, -1]

model = Sequential()

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=100,
        input_length=max_sequence_length - 1
    )
)

model.add(LSTM(100))
model.add(Dense(total_words, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x,
    y,
    epochs=10,
    verbose=1
)