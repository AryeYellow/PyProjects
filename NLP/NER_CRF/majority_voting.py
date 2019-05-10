import pandas as pd
from sklearn.metrics import classification_report

# 数据
df = pd.read_csv('train.csv')
df = df.fillna(method='ffill')

X = df.Word.values
y = df.Tag.values

labels = df.Tag.unique().tolist()
labels.remove('O')


class Majority_vote:
    def fit(self, X, y):
        counter = {}
        for w, t in zip(X, y):
            if w in counter:
                if t in counter[w]:
                    counter[w][t] += 1
                else:
                    counter[w][t] = 1
            else:
                counter[w] = {t: 1}
        self.vote = {}
        for w, t in counter.items():
            self.vote[w] = max(t, key=t.get)
        return self

    def predict(self, X):
        return [self.vote.get(x, 'O') for x in X]


y_pred = Majority_vote().fit(X, y).predict(X)
report = classification_report(y, y_pred, labels)
print(report)