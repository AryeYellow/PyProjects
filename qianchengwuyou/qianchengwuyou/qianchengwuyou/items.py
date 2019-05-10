# -*- coding: utf-8 -*-

import scrapy

class QianchengwuyouItem(scrapy.Item):
    # define the fields for your item here like:
    ls = ['url', 'name', 'salary', 'region', 'workplace',
          'cp_name', 'cp_type', 'cp_scale', 'industry',
          'exp', 'edu', 'demand', 'pubdate', 'skill',
          'welfare',
          'detail',
          'kw', 'collect_date']
    for fd in ls:
        exec(fd + '=scrapy.Field()')
