# -*- coding: utf-8 -*-
import os
import sys

import pkg_resources
get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                            os.path.join(*res))

default_encoding = sys.getfilesystemencoding()

string_types = (str,)

def strdecode(sentence):
    if not isinstance(sentence, str):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence


def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)
