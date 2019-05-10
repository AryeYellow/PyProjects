from scrapy import cmdline
cmdline.execute(['scrapy', 'crawl', 'example'])

"""
CREATE TABLE quanguo(
url CHAR(255) PRIMARY KEY,
name CHAR(255),
salary CHAR(255),
region CHAR(255),
workplace CHAR(255),
cp_name CHAR(255),
cp_type CHAR(255),
cp_scale CHAR(255),
industry CHAR(255),
exp CHAR(255),
edu CHAR(255),
demand CHAR(255),
pubdate CHAR(255),
skill CHAR(255),
welfare CHAR(255),
detail TEXT,
kw CHAR(255) COMMENT '关键词',
collect_date DATE COMMENT '采集日期'
);
"""
