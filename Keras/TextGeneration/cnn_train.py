import numpy as np, os
from collections import Counter
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPool1D, Dense
from TextGeneration.conf import *
from warnings import filterwarnings
filterwarnings('ignore')  # 不打印警告


def preprocessing():
    """语料加载"""
    with open(corpus_path, encoding='utf-8') as f:
        seq_chr = f.read().replace('\n', '')

    """数据预处理"""
    len_seq = len(seq_chr)  # 语料长度
    chr_ls = Counter(list(seq_chr)).most_common(len_chr)
    chr_ls = [i[0] for i in chr_ls]
    chr2id = {c: i for i, c in enumerate(chr_ls)}
    id2chr = {i: c for c, i in chr2id.items()}
    seq_id = [chr2id[c] for c in seq_chr]  # 文字序列 --> 索引序列
    yield len_seq, id2chr, seq_id

    """输入层和标签"""
    reshape = lambda x: np.reshape(x, (-1, window, 1)) / len_chr
    x = [seq_id[i: i + window] for i in range(len_seq - window)]
    x = reshape(x)
    y = [seq_id[i + window] for i in range(len_seq - window)]
    y = to_categorical(y, num_classes=len_chr)
    print('x.shape', x.shape, 'y.shape', y.shape)
    yield reshape, x, y


(len_seq, id2chr, seq_id), (reshape, x, y) = list(preprocessing())

"""建模"""
if os.path.exists(filepath):
    print('load_model')
    model = load_model(filepath)
else:
    print('modeling')
    model = Sequential()
    model.add(Conv1D(filters, kernel_size * 2, padding='same', activation='relu'))
    model.add(MaxPool1D())
    model.add(Conv1D(filters * 2, kernel_size, padding='same', activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(len_chr, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy')


def draw_sample(predictions, temperature):
    """随机采样"""
    pred = predictions.astype('float64')  # 提高精度防报错
    pred = np.log(pred) / temperature
    pred = np.exp(pred)
    pred = pred / np.sum(pred)
    pred = np.random.multinomial(1, pred, 1)
    return np.argmax(pred)


def predict(t, pred=None):
    """预测"""
    if pred is None:
        randint = np.random.randint(len_seq - window)
        pred = seq_id[randint: randint + window]
    if t:
        print('随机采样，温度：%.1f' % t)
        sample = draw_sample
    else:
        print('贪婪采样')
        sample = np.argmax
    for _ in range(window):
        x_pred = reshape(pred[-window:])  # 窗口滑动
        y_pred = model.predict(x_pred)[0]
        i = sample(y_pred, t)
        pred.append(i)
    text = ''.join([id2chr[i] for i in pred[-window:]])
    print('\033[033m%s\033[0m' % text)


"""训练及评估"""
for e in range(times):
    model.fit(x, y, batch_size, epochs, verbose=2)
    model.save(filepath)
    print(str(e + 1).center(window * 2, '-'))
    # 训练效果展示
    for t in (None, 1, 1.5, 2):
        predict(t)
