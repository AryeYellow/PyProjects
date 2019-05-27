from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from os.path import exists

PATH_TRAIN = 'data/train.txt'
PATH_TEST = 'data/test.txt'
PATH_VOCAB = 'model/config.pkl'
PATH_TRAIN_P = 'data/train'

# B表示开始的字节，I表示中间的字节
chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]


def _parse_data(path=PATH_TRAIN):
    with open(path, encoding='utf-8') as f:
        return [sample.split() for sample in f.read().strip().split('\n\n')]


def _pad_sequences(data, maxlen=None):
    maxlen = maxlen or max(len(s) for s in data)
    word2idx = load_vocab()
    x = [[word2idx.get(w.lower(), 0)for w in s[::2]] for s in data]  # 奇数
    y_chunk = [[chunk_tags.index(w)for w in s[1::2]] for s in data]  # 偶数
    # left padding
    x = pad_sequences(x, maxlen)
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def _process_data():
    train = _parse_data()
    print(len(train))
    word_counts = Counter(row.lower() for sample in train for row in sample)
    vocab = [w for w, f in word_counts.most_common() if f >= 2]
    word2idx = {w: i for i, w in enumerate(vocab, 1)}
    print(len(word2idx))
    # 保存
    with open(PATH_VOCAB, 'wb') as f:
        pickle.dump(word2idx, f)
    x_train, y_train = _pad_sequences(train)
    with open(PATH_TRAIN_P, 'wb') as f:
        pickle.dump((x_train, y_train), f)


def load_data(path=PATH_TRAIN_P):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_vocab():
    if not exists(PATH_VOCAB):
        _process_data()
    with open(PATH_VOCAB, 'rb') as f:
        return pickle.load(f)


def pad_seq(seq):
    word2idx = load_vocab()
    x = [word2idx.get(w[0].lower(), 0) for w in seq]
    x = pad_sequences([x], len(x))
    return x


if __name__ == '__main__':
    load_vocab()
