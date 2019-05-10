from time import time
from a_Data_preprocessing import id2label
from sklearn.metrics import classification_report, confusion_matrix
from seaborn import heatmap
from matplotlib import pyplot
from pandas import DataFrame


class Timer:
    def __init__(self):
        self.t = time()

    def __del__(self):
        print('\033[033m%.2f分钟\033[0m' % ((time() - self.t) / 60))


def metric(y_test, y_pred, verbose=True):
    i2l = id2label()
    y_test = [i2l[i] for i in y_test]
    y_pred = [i2l[i] for i in y_pred]
    report = classification_report(y_test, y_pred)
    print(report)
    if verbose:
        labels = [i2l[i] for i in range(9)]
        matrix = confusion_matrix(y_test, y_pred)
        matrix = DataFrame(matrix, labels, labels)
        heatmap(matrix, center=400, fmt='d', annot=True)
        pyplot.show()
