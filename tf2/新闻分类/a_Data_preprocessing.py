import os, pickle


PATH = 'file/'
PATH_DATA = PATH + 'data/'
PATH_DATA2 = PATH + 'data2/'
PATH_SW = PATH + 'stopwords.txt'
PATH_LABEL = PATH_DATA2 + 'id2label'
PATH_XY = PATH_DATA2 + 'xy'
PATH_XY_ID = PATH_DATA2 + 'xy_id'
PATH_XY_VEC = PATH_DATA2 + 'xy_vec'
vocabs = 50000  # vocabulary size
size = 100  # 词向量维度
window = 10  # 词窗大小


def id2label():
    with open(PATH_LABEL, 'rb') as f:
        return pickle.load(f)


def load_xy(i=0):
    path = [PATH_XY, PATH_XY_ID, PATH_XY_VEC]
    with open(path[i], 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import re, jieba
    from gensim.models import FastText
    from b_metrics import Timer
    t = Timer()  # 计时器

    # ID→标签
    labels = {i: l.replace('.txt', '_') for i, l in enumerate(os.listdir(PATH_DATA))}
    with open(PATH_LABEL, 'wb') as f:
        pickle.dump(labels, f)

    # 停词过滤
    with open(PATH_SW, encoding='utf-8') as f:
        stopwords = set(f.read().split())  # 中文停词
        stopwords |= set(ENGLISH_STOP_WORDS)  # 英文停词
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
    X_train, X_test, y_train, y_test = train_test_split(ls_of_words, ls_of_label, test_size=.2)
    del ls_of_words, ls_of_label  # 释放内存

    # 词频统计
    counter = Counter()
    for words in X_train:
        for word in words:
            counter[word] += 1
    counter = counter.most_common(vocabs)
    w2i = {w[0]: i for i, w in enumerate(counter, 1)}

    # 过滤低频词
    X_train = [[w for w in words if w in w2i] for words in X_train]
    X_test = [[w for w in words if w in w2i] for words in X_test]
    y_train = [y for y, x in zip(y_train, X_train) if x]
    y_test = [y for y, x in zip(y_test, X_test) if x]
    X_train = [x for x in X_train if x]
    X_test = [x for x in X_test if x]
    with open(PATH_XY, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    # 转ID
    X_train_i = [[w2i[w] for w in words] for words in X_train]
    X_test_i = [[w2i[w] for w in words] for words in X_test]
    with open(PATH_XY_ID, 'wb') as f:
        pickle.dump((X_train_i, X_test_i, y_train, y_test), f)

    # 词向量
    X = [[str(y)] + x for x, y in zip(X_train, y_train)]
    model = FastText(X, size=size, window=window)
    w2i = {w: i for i, w in enumerate(model.wv.index2word)}
    vectors = model.wv.vectors
    w2v = lambda w: vectors[w2i[w]]
    X_train_v = [[w2v(w) for w in x] for x in X_train]
    X_test_v = [[w2v(w) for w in x if w in w2i] for x in X_test]
    with open(PATH_XY_VEC, 'wb') as f:
        pickle.dump((X_train_v, X_test_v, y_train, y_test), f)
