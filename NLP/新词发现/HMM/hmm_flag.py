import pandas as pd
from jieba3.posseg import cut
from re import sub
from collections import Counter

# 读取jieba默认词典
df = pd.read_table('jieba3/dict.txt', sep=' ', header=None)[[0]]
s = set(df.values.reshape(-1)) | {' '}

# 读取语料
with open('三国演义.txt', encoding='utf-8') as f:
    text = sub('[^\u4e00-\u9fa5]+', ' ', f.read())

# HMM发现新词，并按词频大小降序；存excel
counter = Counter(w for w in cut(text) if w.word not in s)
counter = [(i[0].word, i[0].flag, i[1]) for i in counter.most_common()]
pd.DataFrame(counter, columns=['word', 'flag', 'freq']).\
    to_excel('save/new_word_flag.xlsx', index=False)
