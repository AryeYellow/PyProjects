import pandas as pd, jieba
from os.path import dirname
from re import sub
from collections import Counter

# 读取jieba默认词典
jieba_dict = dirname(jieba.__file__) + '\dict.txt'
df = pd.read_table(jieba_dict, sep=' ', header=None)[[0]]
s = set(df.values.reshape(-1)) | {' '}

# 读取语料
with open('三国演义.txt', encoding='utf-8') as f:
    text = sub('[^\u4e00-\u9fa5]+', ' ', f.read())

# HMM发现新词，并按词频大小降序；存excel
counter = Counter(w for w in jieba.cut(text) if w not in s)
pd.DataFrame(counter.most_common(), columns=['word', 'freq']).\
    to_excel('save/new_word.xlsx', index=False)
