# -*- coding: utf-8 -*-

import pymysql
from time import strftime


class QianchengwuyouPipeline(object):
    def open_spider(self, spider):
        self.db = pymysql.connect('localhost', 'root', 'yellow', charset='utf8', db='z_51job')
        self.cursor = self.db.cursor()

    def close_spider(self, spider):
        self.cursor.close()
        self.db.close()

    def process_item(self, item, spider):
        item['collect_date'] = strftime('%Y-%m-%d')
        ls = [(k, item[k]) for k in item if item[k] is not None]
        sql = 'INSERT quanguo (' + ','.join([i[0] for i in ls]) + \
              ') VALUES (' + ','.join(['%r' % i[1] for i in ls]) + ');'
        print('\033[033m', sql, '\033[0m')
        self.cursor.execute(sql)
        self.db.commit()
        return item

