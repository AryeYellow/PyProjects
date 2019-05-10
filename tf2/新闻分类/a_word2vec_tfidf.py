import os, pickle


PATH = 'file/'
PATH_DATA = PATH + 'data/'
PATH_DATA2 = PATH + 'data2/'
PATH_DATA3 = PATH + 'data3/'
PATH_SW = PATH + 'stopwords.txt'
PATH_LABEL = PATH_DATA2 + 'id2label'
PATH_XY = PATH_DATA3 + 'xy'
PATH_XY_VEC = PATH_DATA3 + 'xy_vec'
PATH_XY_VEC_TFIDF = PATH_DATA3 + 'xy_vec_tfidf'
size = 100  # 词向量维度
window = 10  # 词窗大小


def id2label():
    if not os.path.exists(PATH_LABEL):
        labels = {i: l.replace('.txt', '_') for i, l in enumerate(os.listdir(PATH_DATA))}
        with open(PATH_LABEL, 'wb') as f:
            pickle.dump(labels, f)
    with open(PATH_LABEL, 'rb') as f:
        return pickle.load(f)


def load_xy(i=0):
    path = [PATH_XY_VEC, PATH_XY_VEC_TFIDF]
    if not os.path.exists(PATH_XY_VEC_TFIDF):
        word2vector()
    with open(path[i], 'rb') as f:
        return pickle.load(f)


def text_filter():
    import re, jieba
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS  # 英文停词
    with open(PATH_SW, encoding='utf-8') as f:  # 中文停词
        stopwords = set(f.read().split())
        stopwords |= set(ENGLISH_STOP_WORDS)
    labels = id2label()  # ID→标签
    ls_of_words = []
    ls_of_label = []
    for label_id, label in labels.items():
        with open(PATH_DATA + label.replace('_', '.txt'), encoding='utf-8') as f:
            for line in f.readlines():
                line = ''.join(re.findall('^\[(.+)\]$', line)).strip().lower()
                line = re.sub('\d+[.:年月日时分秒]+', ' ', line)
                line = re.sub('[^\u4e00-\u9fa5a-zA-Z]+', ' ', line)
                words = [w for w in jieba.cut(line) if w.strip() and w not in stopwords]
                if words:
                    ls_of_words.append(words)
                    ls_of_label.append(label_id)
    return train_test_split(ls_of_words, ls_of_label, test_size=.2)


def word2vector():
    from gensim.models import Word2Vec, TfidfModel
    from collections import Counter
    import numpy as np

    X_train, X_test, y_train, y_test = text_filter()

    """词向量"""
    word2vec = Word2Vec(X_train, size, window=window)
    w2i = {w: i for i, w in enumerate(word2vec.wv.index2word)}
    vectors = word2vec.wv.vectors
    # 词→ID
    X_train = [[w2i[w] for w in x if w in w2i] for x in X_train]
    X_test = [[w2i[w] for w in x if w in w2i] for x in X_test]
    # 过滤空值
    y_train = [y for y, x in zip(y_train, X_train) if x]
    y_test = [y for y, x in zip(y_test, X_test) if x]
    X_train = [x for x in X_train if x]
    X_test = [x for x in X_test if x]
    # 保存
    with open(PATH_XY_VEC, 'wb') as f:
        pickle.dump((
            [[vectors[i] for i in x] for x in X_train],
            [[vectors[i] for i in x] for x in X_test],
            y_train, y_test), f)

    """词向量+TfIdf"""
    tfidf = TfidfModel([Counter(x).most_common() for x in X_train])
    idfs = np.array([[tfidf.idfs[i]] for i in range(len(w2i))])
    vectors = vectors * idfs
    # 保存
    with open(PATH_XY_VEC_TFIDF, 'wb') as f:
        pickle.dump((
            [[vectors[i] for i in x] for x in X_train],
            [[vectors[i] for i in x] for x in X_test],
            y_train, y_test), f)


if __name__ == '__main__':
    from b_metrics import Timer
    t = Timer()  # 计时器
    word2vector()
