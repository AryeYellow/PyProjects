import re
import pandas as pd, jieba
from os.path import dirname
from collections import Counter

# 读取jieba默认词典
jieba_dict = dirname(jieba.__file__) + '\dict.txt'
df = pd.read_table(jieba_dict, sep=' ', header=None)[[0]]
s = set(df.values.reshape(-1)) | {' '}

# 读取语料
with open('三国演义.txt', encoding='utf-8') as f:
    text = re.sub('[^\u4e00-\u9fa5]+', ' ', f.read())
le = len(text)

# 遍历
writer = pd.ExcelWriter('save/new_words.xlsx')
for n in (2, 3, 4):
    cn = re.compile('[\u4e00-\u9fa5]{%d}' % n)
    counter = Counter(text[i: i + n] for i in range(le - n)
                      if cn.fullmatch(text[i: i + n]) and
                      text[i: i + n] not in s)
    pd.DataFrame(counter.most_common(9999), columns=['w', 'f'])\
        .to_excel(writer, sheet_name=str(n), index=False)
writer.save()


