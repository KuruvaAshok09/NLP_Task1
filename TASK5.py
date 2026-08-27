import string
import random
import nltk

from nltk import FreqDist, ngrams
from nltk.corpus import stopwords, reuters
from collections import Counter, defaultdict

# Load Reuters sentences
sents = reuters.sents()

# Stopwords and punctuation
stop_words = set(stopwords.words('english'))

punctuation = string.punctuation + '" - _'
removal_list = list(stop_words) + list(punctuation) + ['\t', '\n']

unigram = []
bigram = []
trigram = []

tokenized_text = []

# Process each sentence
for sentence in sents:
    sentence = list(map(lambda x: x.lower(), sentence))

    # Remove periods
    sentence = [word for word in sentence if word != '.']

    # Add words to unigram
    for word in sentence:
        unigram.append(word)

    tokenized_text.append(sentence)

    # Create bigrams
    bigram.extend(
        list(ngrams(sentence, 2, pad_left=True, pad_right=True))
    )

    # Create trigrams
    trigram.extend(
        list(ngrams(sentence, 3, pad_left=True, pad_right=True))
    )


# Remove stopwords and punctuation
def remove_stopwords(x):
    y = []

    for pair in x:
        count = 0

        for word in pair if isinstance(pair, tuple) else [pair]:
            if word in removal_list:
                count = count or 0
            else:
                count = count or 1

        if count == 1:
            y.append(pair)

    return y


unigram = remove_stopwords(unigram)
bigram = remove_stopwords(bigram)
trigram = remove_stopwords(trigram)

# Calculate frequency distributions
freq_unigram = FreqDist(unigram)
freq_bigram = FreqDist(bigram)
freq_trigram = FreqDist(trigram)


# Create trigram dictionary
d = defaultdict(Counter)

for (a, b, c), frequency in freq_trigram.items():
    if a is not None and b is not None and c is not None:
        d[(a, b)][c] += frequency


# Select a random word based on frequency
def pick_word(counter):
    words = list(counter.elements())

    if not words:
        return None

    return random.choice(words)


# Starting prefix
prefix = ("he", "is")

print(" ".join(prefix))

s = " ".join(prefix)

# Generate text
for i in range(9):
    suffix = pick_word(d[prefix])

    if suffix is None:
        break

    s = s + " " + suffix
    print(s)

    prefix = (prefix[1], suffix)