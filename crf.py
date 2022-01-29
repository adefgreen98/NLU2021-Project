import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


from utils import get_dataset

import spacy
import en_core_web_lg

nlp = en_core_web_lg.load()

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


train_dataset = get_dataset('iob_atis/atis.train.pkl')
test_dataset = get_dataset('iob_atis/atis.test.pkl')

X_train, y_train = [], []
for x,y in train_dataset:
    parse = nlp(' '.join(x))
    sent = [[word.text, word.tag_, lab] for word, lab in zip(parse, y)]
    X_train.append(sent2features(sent))
    y_train.append(sent2labels(sent))

X_test, y_test = [], []
for x,y in test_dataset:
    parse = nlp(' '.join(x))
    sent = [[word.text, word.tag_, lab] for word, lab in zip(parse, y)]
    X_test.append(sent2features(sent))
    y_test.append(sent2labels(sent))

print("Before fit...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')

print("Predicting...")
y_pred = crf.predict(X_test)

# group B and I results
sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)

from utils import conll_evaluate

print(conll_evaluate([el for yt in y_test for el in yt], [el for yp in y_pred for el in yp]), verbose=True)

# (88.8949522510232, 95.93954843408595, 90.379113018598, 89.07296439901305, 89.72128528315285)