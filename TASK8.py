import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.classify import MaxentClassifier

nltk.download('maxent_treebank_pos_tagger')

from nltk.tag import PerceptronTagger, StanfordTagger

nltk.download('treebank')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

corpus = list(treebank.tagged_sents())

train_data = corpus[:int(0.8 * len(corpus))]
test_data = corpus[int(0.8 * len(corpus)):]

hmm_tagger = hmm.HiddenMarkovModelTrainer().train(train_data)

hmm_accuracy = hmm_tagger.accuracy(test_data)

print(f"HMM Tagger Accuracy: {hmm_accuracy:.4f}")