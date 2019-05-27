from numpy import argmax
from bilsm_crf_model import modeling
from process_data import pad_seq, chunk_tags

predict_text = '国务院总理在外交部长陈毅陪同下，访问了埃塞俄比亚'
# predict_text = '国家主席习近平向印尼总统佐科·维多多致贺电，代表中国政府和中国人民，并以个人的名义祝贺他胜选连任印度尼西亚总统。'

model = modeling(train=False)
x = pad_seq(predict_text)
raw = model.predict(x[0])[-len(predict_text):]
print(raw)
print(raw.shape)
result_tags = [chunk_tags[argmax(row)] for row in raw[:, 0]]

per, loc, org = '', '', ''
for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print('person:' + per, 'location:' + loc, 'organzation:' + org)
