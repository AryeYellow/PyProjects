"""基于jieba改写的中文分词算法"""
import os, re, pandas as pd, pickle
from math import log
from time import time
# 基础目录
BASE_PATH = os.path.dirname(__file__)
# 生成绝对路径
get_abs_path = lambda path1, path2: os.path.normpath(os.path.join(path1, path2))
# 通用词库
JIEBA_DICT = get_abs_path(BASE_PATH, 'jieba_dict.txt')  # jieba原词典
NA = 'NA'


class Io:
    """读文件"""
    @staticmethod
    def txt2df(filename=JIEBA_DICT, sep=' ', names=None):
        return pd.read_table(filename, sep, names=names, header=None)

    @staticmethod
    def xlsx2df(filename, sheet_name=0):
        return pd.read_excel(filename, sheet_name=sheet_name)

    @staticmethod
    def csv2df(filename, header='infer', names=None):
        return pd.read_csv(filename, header=header, names=names)

    @staticmethod
    def pickle2dt(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    """数据处理"""
    @staticmethod
    def df2dt(df):
        if df.shape[1] == 2:
            return dict(df.values)
        elif df.shape[1] > 2:
            return {i[0]: i[1:] for i in df.values}

    @staticmethod
    def concat(ls_of_df):
        """合并DataFrame，按第0列去重，保留前者"""
        df = pd.concat(ls_of_df)
        df.drop_duplicates(subset=df.columns[0], inplace=True)
        return df
    """写文件"""
    @staticmethod
    def df2csv(filename, df):
        df.to_csv(filename, index=False)

    @staticmethod
    def dt2pickle(filename, dt):
        with open(filename, 'wb') as f:
            pickle.dump(dt, f)


class Word:
    def __init__(self, word, flag=NA, value=0, **kwargs):
        self.word = word  # str
        self.flag = flag  # str
        self.value = str(value)  # int or float
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.word

    def __repr__(self):
        return repr(self.__dict__)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        """判断flag是否相等"""
        if isinstance(other, self.__class__):
            return self.flag == other.flag
        elif isinstance(other, str):
            return self.flag == other
        else:
            return False

    def __iadd__(self, other):
        """
        +=
            美丽 + 动人 = 美丽动人
        """
        assert self == other, '对象类型不匹配'
        self.word += other.word
        self.value = str(int(self.value) + int(other.value))
        return self

    def __imul__(self, other):
        """
        *=
            很 * tm = 很tm
        """
        assert self == other, '对象类型不匹配'
        self.word += other.word
        self.value = str(int(self.value) * int(other.value))
        return self

    def __ior__(self, other):
        """
        |= 【位或；并集】
            酸奶和芝士 = 酸奶|芝士
        """
        assert self == other, '对象类型不匹配'
        self.word = self.word + '|' + other.word
        self.value = self.value + '|' + other.value
        return self

    def __and__(self, other):
        """
        & 【位与；交集】
            很 & 不错 = 很不错
        """
        assert isinstance(other, self.__class__), '对象类型不匹配'
        self.word += other.word
        self.flag = other.flag
        self.value = str(int(self.value) * int(other.value))
        return self


class Cutter:
    re_eng = re.compile('[a-zA-Z0-9_\-]+')
    re_m = re.compile('[0-9.\-+%/~]+')  # jieba数词标注为m

    def __init__(self, dt, total, max_len):
        self.t = time()
        self.dt = dt
        self.total = total
        self.max_len = max_len

    def __del__(self):
        t = time() - self.t
        print('分词耗时：%.2f秒' % t) if t < 60 else print('分词耗时：%.2f分钟' % (t/60))

    @classmethod
    def initialization(cls):
        df = Io.txt2df()
        dt = Io.df2dt(df)
        # 总频数
        total = df[1].sum()
        # 词最大长度，默认等于词典最长词
        max_len = df[0].str.len().max()
        return cls(dt, total, max_len)

    def _get_DAG(self, sentence):
        length = len(sentence)
        dt = dict()
        for head in range(length):
            tail = min(head + self.max_len, length)
            dt.update({head: [head]})
            for middle in range(head + 2, tail + 1):
                word = sentence[head: middle]
                # ------------- 词典 + 正则 ------------- #
                if word in self.dt:
                    dt[head].append(middle - 1)
                elif self.re_eng.fullmatch(word):
                    dt[head].append(middle - 1)
                    self.add_word(word, 1, 'eng')
                elif self.re_m.fullmatch(word):
                    dt[head].append(middle - 1)
                    self.add_word(word, 1, 'm')  # jieba数词m
        return dt

    def _calculate(self, sentence):
        DAG = self._get_DAG(sentence)
        route = {}
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (log(self.dt.get(sentence[idx:x + 1], (1,))[0]) - logtotal + route[x + 1][0], x)
                for x in DAG[idx])
        return route

    def cut(self, sentence):
        route = self._calculate(sentence)
        x = 0
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            yield l_word
            x = y

    def lcut(self, sentence):
        return list(self.cut(sentence))

    def add_word(self, word, freq=1, flag=NA):
        original_freq = self.dt.get(word, (0,))[0]
        self.dt[word] = [freq, flag]
        self.total = self.total - original_freq + freq

    def del_word(self, word):
        original_freq = self.dt.get(word)
        if original_freq:
            del self.dt[word]
            self.total -= original_freq[0]

    def cut2w(self, sentence):
        for word in self.cut(sentence):
            yield Word(word, self.dt.get(word, [0, NA])[1])

    def lcut2w(self, sentence):
        return [w for w in self.cut2w(sentence)]

    def cut2position(self, sentence):
        route = self._calculate(sentence)
        x = 0
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            yield l_word, (x, y - 1)
            x = y


cutter = Cutter.initialization
cut = lambda sentence: cutter().cut(sentence)
lcut = lambda sentence: cutter().lcut(sentence)


if __name__ == '__main__':
    c = cutter()
    print(lcut('Angelababy抱儿子睡觉画面温馨'))
    print(list(c.cut2position('Facebook总部大楼被疏散')))
    c.add_word('米家', 99, 'nt')
    print(c.lcut2w('米家APP升级'))
