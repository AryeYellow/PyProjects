# jieba简化

## 全局删
- PY2
- \_\_future\_\_
- (object)

## 删除\_\_main\_\_.py文件

## 修改_compat.py
- 删try
- text_type全局替换
- string_types全局替换
- xrange全局替换
- iterkeys全局替换
- itervalues全局替换
- iteritems全局替换
- 删sys
- strdecode全局删
- pkg_resources全局删
- resolve_filename

## 修改__init__.py
- 删\_\_version\_\_、\_\_license\_\_
- 删logging
- 删time
- 删threading
- 删pool

## finalseg
### 删除所有.p文件
### 修改__init__.py
- 删.p文件名、load_model
- Force_Split_Words = set()


## posseg
### 删除所有.p文件
### 修改__init__.py
- 删.p文件名、load_model


# 自写算法，加入jieba

