import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('stopwords')

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file: 
        text = file.read()
    return text

def tokenize_document(document):
    tokens = word_tokenize(document)
    return [word.lower() for word in tokens if word.isalpha()]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def find_morphology(tokens):
    freq_dist = FreqDist(tokens)
    most_common_words = freq_dist.most_common(10)
    return most_common_words

document_path = 'document.txt'

document = load_document(document_path)
tokens = tokenize_document(document)
tokens_without_stopwords = remove_stopwords(tokens)
morphology = find_morphology(tokens_without_stopwords)

print('Most Frequent Words in the Document')
for word, frequency in morphology:
    print(f"{word}: {frequency}")