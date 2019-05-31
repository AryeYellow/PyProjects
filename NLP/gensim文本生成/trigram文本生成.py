from collections import Counter
from jieba import lcut
from os.path import exists
import pickle
from random import choice

PATH1 = 'unigram'
PATH2 = 'bigram'
PATH3 = 'trigram'

# N-gram建模训练
if not exists(PATH2) or not exists(PATH3) or not exists(PATH3):
    with open('corpus.txt', encoding='utf-8') as f:
        corpus = [lcut(line) for line in f.read().strip().split()]

if not exists(PATH1):
    unigram = Counter(word for words in corpus for word in words)
    with open(PATH1, 'wb') as f:
        pickle.dump(unigram, f)
with open(PATH1, 'rb') as f:
    unigram = pickle.load(f)

if not exists(PATH2):
    bigram = {w: Counter() for w in unigram.keys()}
    for words in corpus:
        for i in range(len(words) - 1):
            bigram[words[i]][words[i + 1]] += 1
    with open(PATH2, 'wb') as f:
        pickle.dump(bigram, f)
with open(PATH2, 'rb') as f:
    bigram = pickle.load(f)

if not exists(PATH3):
    trigram = {words[i]+words[i+1]: Counter() for words in corpus for i in range(len(words) - 1)}
    for words in corpus:
        for i in range(len(words) - 2):
            trigram[words[i]+words[i+1]][words[i+2]] += 1
    with open(PATH3, 'wb') as f:
        pickle.dump(trigram, f)
with open(PATH3, 'rb') as f:
    trigram = pickle.load(f)

# 文本生成
n = 9  # 开放度
while True:
    first = input('首字：').strip()
    if first not in unigram:
        first = choice(list(unigram))
    second = sorted(bigram[first], key=lambda w: bigram[first][w])[:n]
    second = choice(second) if second else ''
    next_word = sorted(trigram[first+second], key=lambda w: trigram[first+second][w])[:n]
    next_word = choice(next_word) if next_word else ''
    sentence = [first, next_word]
    next_word = first + next_word
    for i in range(99):
        if next_word in trigram:
            next_word = sorted(trigram[next_word], key=lambda w: trigram[next_word][w])
            if next_word:
                next_word = choice(next_word)
            else:
                next_word = sentence[-1]
                next_word = sorted(bigram[next_word], key=lambda w: bigram[next_word][w])[:n]
                if next_word:
                    next_word = choice(next_word)
                else:
                    break
        sentence.append(next_word)
        next_word = ''.join(sentence[-2:])
        print(i, next_word)
    print(''.join(sentence))
