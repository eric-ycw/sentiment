from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
from collections import Counter


def create_lexicon(data_file):
    print('Creating lexicon...')
    lexicon = []
    with open(data_file, 'r') as f:
        for l in f.readlines():
            lexicon += list(word_tokenize(l.lower()))

    wnl = WordNetLemmatizer()
    lexicon = [wnl.lemmatize(w) for w in lexicon]

    word_counts = Counter(lexicon)
    lexicon = [w for w in word_counts if w not in stopwords.words('english')]
    return [w for w in lexicon if word_counts[w] > 8]


def process_features(data_file, lexicon):
    print('Processing data...\n')
    feature_set = []
    with open(data_file, 'r') as f:
        for line_num, l in enumerate(f.readlines()):
            if line_num % 500 == 0:
                print('Reading line', line_num)

            # Note: Must leave one blank line at the end of the data file
            l = l.rstrip('\n')
            if l[-1:].isdigit():
                classification = int(l[-1:])
            else:
                continue

            wnl = WordNetLemmatizer()
            words = [wnl.lemmatize(w) for w in word_tokenize(l.lower())]

            features = np.zeros(len(lexicon))
            for w in words:
                if w in lexicon:
                    features[lexicon.index(w)] += 1
            if classification:
                feature_set.append([features, [1, 0]])
            else:
                feature_set.append([features, [0, 1]])

    return feature_set


def create_features_and_labels(data_file, test_factor=0.1, lexicon=None):
    if lexicon is None:
        lexicon = create_lexicon(data_file)
    features = []
    features += process_features(data_file, lexicon)
    random.shuffle(features)
    features = np.array(features)

    test_size = int(test_factor * len(features))
    x_train, y_train = list(features[:, 0][:-test_size]), list(features[:, 1][:-test_size])
    x_test, y_test = list(features[:, 0][-test_size:]), list(features[:, 1][-test_size:])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)