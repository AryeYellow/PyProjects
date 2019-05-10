# -*- coding: utf-8 -*-
import scrapy
from urllib import parse
from ..items import QianchengwuyouItem
import re
kw = input('输入搜索词：').strip()


class ExampleSpider(scrapy.Spider):
    name = 'example'
    # 起始页，两层URL编码
    start_urls = ['https://search.51job.com/list/000000,000000,0000,00,9,99,{},2,1.html'.format(
        parse.quote(parse.quote(kw)))]
    # 搜索页
    def parse(self, response):
        ls_url = response.xpath('//p/span/a').re('href="(.*?)"')
        for url in ls_url:
            yield scrapy.Request(url=url, callback=self.parse_detail)
        next_page = response.xpath('//*[@id="resultList"]/div[55]/div/div/div/ul/li[8]/a').re('href="(.*?)"')
        if next_page:
            yield scrapy.Request(url=next_page[0], callback=self.parse)
    # 详情页
    def parse_detail(self, response):
        item = QianchengwuyouItem()
        # 基本信息
        item['kw'] = kw
        item['name'] = response.xpath('/html/body/div[3]/div[2]/div[2]/div/div[1]/h1/@title').extract_first()
        item['url'] = response.url
        item['salary'] = ''.join(response.xpath('/html/body/div[3]/div[2]/div[2]/div/div[1]/strong/text()').extract())
        region_exp_edu = response.xpath('/html/body/div[3]/div[2]/div[2]/div/div[1]/p[2]/@title').extract_first().split('\xa0\xa0|\xa0\xa0')
        item['region'] = region_exp_edu[0]
        item['cp_name'] = response.xpath('/html/body/div[3]/div[2]/div[2]/div/div[1]/p[1]/a[1]/@title').extract_first()
        item['workplace'] = ''.join(response.xpath('/html/body/div[3]/div[2]/div[3]/div[2]/div/p').re('</span>(.+?)</p>')).strip()
        item['welfare'] = '|'.join(response.xpath('/html/body/div[3]/div[2]/div[2]/div/div[1]/div/div/span/text()').extract())
        # 详情页
        detail = ''.join(response.xpath('/html/body/div[3]/div[2]/div[3]/div[1]/div').extract()).lower()
        detail = re.sub('<[^>]*>', '', detail)
        item['detail'] = re.sub("""[^，。？！；：‘’“”、【】…0-9.\-a-zA-Z\u4e00-\u9fa5]+""", '', detail)
        # 经验、学历、招聘人数、发布日期
        item['exp'] = item['edu'] = item['demand'] = item['pubdate'] = item['skill'] = ''
        EDU = ['博士', '硕士', '本科', '大专',
               '中专', '中技', '高中', '初中及以下']
        for i in region_exp_edu:
            if '经验' in i:
                item['exp'] = i
            elif i in EDU:
                item['edu'] = i
            elif '招' in i:
                item['demand'] = i
            elif '发布' in i:
                item['pubdate'] = i
            else:
                item['skill'] = i
        # 公司信息
        CP_TYPE = ['民营公司', '上市公司', '事业单位', '国企', '外资（欧美）', '外资（非欧美）',
                   '创业公司', '政府机关', '合资', '外资', '合资', '外企代表处', '非营利组织']
        CP_SCALE = ['少于50人', '50-150人', '150-500人', '500-1000人',
                    '1000-5000人', '5000-10000人', '10000人以上']
        cp_info = response.xpath('/html/body/div[3]/div[2]/div[4]/div[1]/div[2]/p/text()').extract()
        cp_info = [i.strip() for i in cp_info if i.strip()]
        item['cp_type'] = item['cp_scale'] = item['industry'] = ''
        for i in CP_TYPE:
            if i in cp_info:
                item['cp_type'] = i
                break
        for i in CP_SCALE:
            if i in cp_info:
                item['cp_scale'] = i
                break
        for i in cp_info:
            if i not in CP_TYPE and i not in CP_SCALE:
                item['industry'] = i
        return item