import nltk
nltk.download('all')
from nltk.tag import HiddenMarkovModelTagger
from nltk.corpus import treebank
nltk.download('treebank')
corpus = treebank.tagged_sents()
train_data = corpus[:3000]
test_data = corpus[3000:]
hmm_tagger = HiddenMarkovModelTagger.train(train_data)
sentence = input()
tokens = nltk.word_tokenize(sentence)
tagged_sentence = hmm_tagger.tag(tokens)
print(tagged_sentence)