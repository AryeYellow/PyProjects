import os
import re
import marshal
import tempfile
from math import log
from hashlib import md5
from . import finalseg

from shutil import move

import pkg_resources
get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                            os.path.join(*res))

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"


DICT_WRITING = {}


class Tokenizer:

    def __init__(self):
        self.dictionary = None
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def gen_pfdict(self, f):
        lfreq = {}
        ltotal = 0
        f_name = f.name
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode('utf-8')
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in range(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()
        return lfreq, ltotal

    def initialize(self, dictionary=None):
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        try:
            with DICT_WRITING[abs_path]:
                pass
        except KeyError:
            pass
        if self.initialized:
            return

        if self.cache_file:
            cache_file = self.cache_file
        # default dictionary
        elif abs_path == DEFAULT_DICT:
            cache_file = "jieba3.cache"
        # custom dictionary
        else:
            cache_file = "jieba3.u%s.cache" % md5(
                abs_path.encode('utf-8', 'replace')).hexdigest()
        cache_file = os.path.join(
            self.tmp_dir or tempfile.gettempdir(), cache_file)
        # prevent absolute path in self.cache_file
        tmpdir = os.path.dirname(cache_file)

        load_from_cache_fail = True
        if os.path.isfile(cache_file) and (abs_path == DEFAULT_DICT or
            os.path.getmtime(cache_file) > os.path.getmtime(abs_path)):
            try:
                with open(cache_file, 'rb') as cf:
                    self.FREQ, self.total = marshal.load(cf)
                load_from_cache_fail = False
            except:
                load_from_cache_fail = True

        if load_from_cache_fail:
            self.FREQ, self.total = self.gen_pfdict(self.get_dict_file())
            try:
                # prevent moving across different filesystems
                fd, fpath = tempfile.mkstemp(dir=tmpdir)
                with os.fdopen(fd, 'wb') as temp_cache_file:
                    marshal.dump(
                        (self.FREQ, self.total), temp_cache_file)
                move(fpath, cache_file)
            except:
                pass

            try:
                del DICT_WRITING[abs_path]
            except KeyError:
                pass

        self.initialized = True

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def get_DAG(self, sentence):
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def __cut_DAG(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
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
                        yield buf
                        buf = ''
                    else:
                        if not self.FREQ.get(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence):
        re_han = re.compile('[\u4E00-\u9FD5a-zA-Z0-9+#&._%]+')
        re_skip = re.compile('\s')

        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                for word in self.__cut_DAG(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    else:
                        for xx in x:
                            yield xx

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def get_dict_file(self):
        return get_module_res(DEFAULT_DICT_NAME)


dt = Tokenizer()
cut = dt.cut
lcut = dt.lcut
