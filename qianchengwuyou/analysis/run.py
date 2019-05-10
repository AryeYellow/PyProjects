import os, sys
# 配置环境变量
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_PATH)
from core.db import Mysql
from core.df import Df
# 执行
if __name__ == '__main__':
    kw = '网络工程'
    db = Mysql()
    df = Df(db.select_key(kw), kw=kw)
    df.write()
