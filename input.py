from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


def process_input(str, lexicon):
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(w) for w in word_tokenize(str.lower())]

    feature_set = []
    features = np.zeros(len(lexicon))
    for w in words:
        if w in lexicon:
            features[lexicon.index(w)] += 1
    feature_set.append([features, None])
    feature_set = np.array(feature_set)
    feature_set = list(feature_set[:, 0][:])

    return np.array(feature_set)