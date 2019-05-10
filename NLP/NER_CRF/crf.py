import pandas as pd
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

data = pd.read_csv('train.csv').fillna(method="ffill")

labels = data.Tag.unique().tolist()
labels.remove('O')

f = lambda s: [(w, p, t) for w, p, t in zip(
    s.Word.values, s.POS.values, s.Tag.values)]
sentences = list(data.groupby("Sentence #").apply(f))


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
    if i > 0: # word_prev, word_curr
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


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

crf = CRF(algorithm='lbfgs', c1=.1, c2=.1, max_iterations=99).fit(X, y)
y_pred = crf.predict(X)
report = flat_classification_report(y, y_pred, labels)
print(report)