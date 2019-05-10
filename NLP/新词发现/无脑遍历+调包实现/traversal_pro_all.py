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
writer = pd.ExcelWriter('save/new_word_pro_all.xlsx')
counter = Counter()
for n in (6, 5, 4, 3, 2):
    cn = re.compile('[\u4e00-\u9fa5]{%d}' % n)
    mc = Counter(text[i: i + n] for i in range(le - n)
                 if cn.fullmatch(text[i: i + n]) and
                 text[i: i + n] not in s).most_common(int(9999 / n))  # 词越长越少
    for word, freq in mc:  # 遍历短词
        for w, f in counter.most_common():  # 遍历长词
            if word in w and\
                    f / freq > .9 ** (len(w) - len(word)):  # 词长差距
                print(w, word, f / freq)
                break  # 短词被包含于长词且词频相近时，不加入新词
        else:
            counter[word] = freq
    # 保存sheet
    pd.DataFrame(counter.most_common(999), columns=['w', 'f'])\
        .to_excel(writer, sheet_name=str(n), index=False)
writer.save()