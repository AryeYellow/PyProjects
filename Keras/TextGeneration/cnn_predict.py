import numpy as np
from collections import Counter
from keras.models import load_model
from warnings import filterwarnings
from TextGeneration.conf import corpus_path, len_chr, window, filepath
filterwarnings('ignore')  # 不打印警告


def draw_sample(predictions, temperature):
    pred = predictions.astype('float64')  # 提高精度防报错
    pred = np.log(pred) / temperature
    pred = np.exp(pred)
    pred = pred / np.sum(pred)
    pred = np.random.multinomial(1, pred, 1)
    return np.argmax(pred)


def preprocessing():
    # 语料加载
    with open(corpus_path, encoding='utf-8') as f:
        seq_chr = f.read().replace('\n', '')
    # 语料处理
    len_seq = len(seq_chr)
    chr_ls = Counter(list(seq_chr)).most_common(len_chr)
    chr_ls = [i[0] for i in chr_ls]
    # 字符和索引间映射
    chr2id = {c: i for i, c in enumerate(chr_ls)}
    id2chr = {i: c for c, i in chr2id.items()}
    c2i = lambda c: chr2id.get(c, np.random.randint(len_chr))
    # 序列处理
    seq_id = [chr2id[c] for c in seq_chr]
    reshape = lambda x: np.reshape(x, (-1, window, 1)) / len_chr
    return len_seq, id2chr, seq_id, c2i, reshape


# 语料加载和处理
len_seq, id2chr, seq_id, c2i, reshape = preprocessing()

# 模型加载
model = load_model(filepath)


def predict(t, pred):
    if t:
        print('随机采样，温度：%.1f' % t)
        sample = draw_sample
    else:
        print('贪婪采样')
        sample = np.argmax
    for _ in range(window):
        x_pred = reshape(pred[-window:])
        y_pred = model.predict(x_pred)[0]
        i = sample(y_pred, t)
        pred.append(i)
    text = ''.join([id2chr[i] for i in pred[-window:]])
    print('\033[033m%s\033[0m' % text)


if __name__ == '__main__':
    for t in (None, 1, 1.5, 2):
        predict(t, seq_id[-window:])
    while True:
        title = input('输入标题').strip() + '。'
        len_t = len(title)
        randint = np.random.randint(len_seq - window + len_t)
        randint = int(randint // 12 * 12)
        pred = seq_id[randint: randint + window - len_t] + [c2i(c) for c in title]
        for t in (None, 1, 1.5, 2):
            predict(t, pred)
