import re
import jieba3
from .viterbi import viterbi


re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile("([.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&._]+)")
re_skip_internal = re.compile("(\r\n|\s)")

re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[.0-9]+")


from .char_state_tab import P as char_state_tab_P
from .prob_start import P as start_P
from .prob_trans import P as trans_P
from .prob_emit import P as emit_P


class Pair:

    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __repr__(self):
        return 'Pair(%r, %r)' % (self.word, self.flag)

    def __eq__(self, other):
        return isinstance(other, Pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)


class POSTokenizer:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or jieba3.Tokenizer()
        self.load_word_tag(self.tokenizer.get_dict_file())

    def initialize(self, dictionary=None):
        self.tokenizer.initialize(dictionary)
        self.load_word_tag(self.tokenizer.get_dict_file())

    def load_word_tag(self, f):
        self.word_tag_tab = {}
        f_name = f.name
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()

    def makesure_userdict_loaded(self):
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence):
        prob, pos_list = viterbi(
            sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield Pair(sentence[begin:i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == 'S':
                yield Pair(char, pos_list[i][1])
                nexti = i + 1
        if nexti < len(sentence):
            yield Pair(sentence[nexti:], pos_list[nexti][1])

    def __cut_detail(self, sentence):
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                for word in self.__cut(blk):
                    yield word
            else:
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            yield Pair(x, 'm')
                        elif re_eng.match(x):
                            yield Pair(x, 'eng')
                        else:
                            yield Pair(x, 'x')

    def __cut_DAG(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}

        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield Pair(buf, self.word_tag_tab.get(buf, 'x'))
                    elif not self.tokenizer.FREQ.get(buf):
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield Pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                yield Pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            x = y

        if buf:
            if len(buf) == 1:
                yield Pair(buf, self.word_tag_tab.get(buf, 'x'))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield Pair(elem, self.word_tag_tab.get(elem, 'x'))

    def __cut_internal(self, sentence):
        self.makesure_userdict_loaded()
        blocks = re_han_internal.split(sentence)
        cut_blk = self.__cut_DAG

        for blk in blocks:
            if re_han_internal.match(blk):
                for word in cut_blk(blk):
                    yield word
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield Pair(x, 'x')
                    else:
                        for xx in x:
                            if re_num.match(xx):
                                yield Pair(xx, 'm')
                            elif re_eng.match(x):
                                yield Pair(xx, 'eng')
                            else:
                                yield Pair(xx, 'x')

    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    def cut(self, sentence):
        for w in self.__cut_internal(sentence):
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))


def cut(sentence):
    dt = POSTokenizer(jieba3.dt)
    dt.initialize()
    return dt.cut(sentence)


def lcut(sentence):
    return list(cut(sentence))
