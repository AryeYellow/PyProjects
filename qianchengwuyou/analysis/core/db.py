import pymysql, pandas as pd


class Mysql:
    # 连接数据库,选择库
    def __init__(self):
        self.db = pymysql.connect('localhost', 'root', 'yellow', charset='utf8', db='z_51job')
        self.cursor = self.db.cursor()
    # 执行SQL
    def execute(self, sentence, arg=None):
        try:
            if arg:
                self.cursor.execute(sentence, arg)
            else:
                self.cursor.execute(sentence)
            self.db.commit()
        except Exception as error:
            print('\033[033m', error, '\033[0m')
    # 提交SQL
    def close(self):
        self.cursor.close()
        self.db.close()
    # 查询主要字段，排除薪资为空的记录
    def select_key(self, kw):
        query = '''select
        name,salary,region,workplace,cp_name,cp_type,cp_scale,industry,exp,edu,demand,pubdate,welfare,detail
        from quanguo where salary != '' and kw = "{}";'''.format(kw)
        df = pd.read_sql(query, self.db)
        return df
