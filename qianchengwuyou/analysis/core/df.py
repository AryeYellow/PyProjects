import re, jieba.posseg as jp, pandas as pd
from collections import Counter
from core.conf import PREFIX, stopwords
# 数据分析
class Df:
    def __init__(self, data_frame, kw='python'):
        self.df = data_frame
        self.writer = pd.ExcelWriter(PREFIX + kw + '.xlsx')
    # 新增百分比列，并保存sheet
    def percentage(self, ls_tup, name):
        fields = [name, 'frequency']
        df = pd.DataFrame(ls_tup, columns=fields)
        df['percentage'] = df[df.columns[1]]/self.df.shape[0]
        df.sort_values(df.columns[1], ascending=False)
        df.to_excel(self.writer, sheet_name=name, index=False)
    # 分析并保存
    def write(self):
        # 中英文分词
        self.cut('detail')
        self.english('detail')
        # 薪资离散化、区间化
        self.salary()
        self.df.to_excel(self.writer, sheet_name='origin', index=False)
        self.pivot('salary_section', 'exp')
        self.pivot('salary_section', 'edu')
        self.pivot('salary_section', 'cp_scale')
        self.pivot('salary_section', 'cp_type')
        self.pivot('salary_section', 'region')
        self.spt('region')
        # self.spt('industry')
        self.spt('welfare')
        self.spt('cp_name')
        self.writer.save()
    # 分词（detail）
    def cut(self, field):
        text = '|'.join([str(i) for i in self.df[field]])  # 衔接文本
        counter = Counter()
        posseg = jp.cut(text)
        for p in posseg:
            if len(p.word) > 2 and p.flag != 'eng' and p.flag != 'm':
                counter[p.flag + ' | ' + p.word] += 1
        most = counter.most_common()
        # 保存
        self.percentage(most, 'Cn%s' % field)
    # 英文分词（detail）
    def english(self, field):
        text = '|'.join([str(i) for i in self.df[field]])  # 衔接文本
        pat = '[a-z]+'
        re_ls = re.findall(pat, text)
        counter = Counter(re_ls)
        c1 = counter.most_common(199)
        # 英文停词过滤
        c2 = [i for i in c1 if i[0] not in stopwords]
        # 保存
        self.percentage(c2, 'En%s' % field)
    # 切割（workplace，industry）
    def spt(self, field):
        text = '|'.join([str(i) for i in self.df[field]])  # 衔接文本
        ls = text.split('|')
        counter = Counter(ls)
        c = counter.most_common()
        # 保存
        self.percentage(c, 'CNT%s' % field)
    # 透视表(薪资分布)
    def pivot(self, field_1, field_2):
        name = field_1 + '-' + field_2
        pivot = self.df.pivot_table(
            values='name',
            index=field_1,
            columns=field_2,
            aggfunc='count')
        pivot.to_excel(
            self.writer,
            sheet_name=name)
    # salary
    def salary(self, n=4, start=5000, end=30000, step=1000):
        salary = self.df['salary']
        salary = ['0-0千/月' if i == '' else i for i in salary]
        # 切割“/”
        spt_ls = [i.split('/') for i in salary]
        # 年月→乘数
        period = {
            '年': 1/12,
            '月': 1,
            '天': 20,
            '小时': 160}
        for spt in spt_ls:
            spt[1] = period[spt[1]]
        # 万千→乘数
        units = {
            '万': 10000,
            '千': 1000,
            '元': 1}
        for spt in spt_ls:
            for k in units.keys():
                if k in spt[0]:
                    spt[0] = spt[0].replace(k, '')
                    spt.append(units[k])
                    break
        # 最大值，最小值
        for spt in spt_ls:
            if '-' in spt[0]:
                mi, ma = spt[0].split('-')
                spt[0] = ma
                spt.insert(0, mi)
            elif '以上' in spt[0]:
                mi = spt[0].split('以上')[0]
                spt[0] = mi
                spt.insert(0, mi)
            elif '以下' in spt[0]:
                mi = spt[0].split('以下')[0]
                spt[0] = mi
                spt.insert(0, mi)
            else:
                spt.insert(0, spt[0])
        # 文本转浮点数
        dfs = pd.DataFrame(spt_ls, columns=['min', 'max', 'period', 'unit'])
        row_min = list(pd.to_numeric(dfs['min']) * dfs['period'] * dfs['unit'])
        row_max = list(pd.to_numeric(dfs['max']) * dfs['period'] * dfs['unit'])
        # 升采样
        row_ls = [row_min, row_max]
        for j in range(1, n):
            row = [(row_min[i] + (row_max[i] - row_min[i]) * j / n) for i in range(len(spt_ls))]
            row_ls.append(row)
        # 重构DataFrame
        df_ls = [pd.concat([self.df, pd.DataFrame(row, columns=['salary_value'])], axis=1) for row in row_ls]
        # 合并DataFrame、升序
        df = pd.concat(df_ls)
        df.sort_values(by=['name', 'salary_value'], axis=0, ascending=True, inplace=True)
        # 薪资区间化（数据离散化）
        bin = [-1, 0] + list(range(start, end, step)) + [999999]  # 自定义区间划分
        open_left = ['(%d,%d]' % (bin[i], bin[i + 1]) for i in range(len(bin) - 1)]  # 长度要-1
        sec = pd.cut(df['salary_value'], bin, labels=open_left).to_frame(name='salary_section')
        # 合并
        self.df = pd.concat([df, sec], axis=1)
