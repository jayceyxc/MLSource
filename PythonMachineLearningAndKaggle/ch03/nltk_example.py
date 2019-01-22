#!/usr/bin/env python3
# @Time    : 2018/10/7 10:24 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : nltk_example.py
# @Software: PyCharm
# @Description NLTK库示例

from sklearn.feature_extraction.text import CountVectorizer
import nltk

# 使用词袋法（Bag-of-Words）对示例文本进行特征向量化
sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'
count_vec = CountVectorizer()

sentences = [sent1, sent2]

# 输出特征向量化后的表示
print(count_vec.fit_transform(sentences).toarray())

# 输出向量各个维度的特征含义
print(count_vec.get_feature_names())

# 使用NLTK对示例文本进行语言学分析
# 对句子进行词汇分割和正规化，有些情况如aren't需要分割为are和n't；或者I'm要分割为I和'm
tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)

tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)

# 整理两句的词表，并且按照ASCII的排序输出
vocab_1 = sorted(set(tokens_1))
print(vocab_1)

vocab_2 = sorted(set(tokens_2))
print(vocab_2)

# 初始化stemmer寻找各个词汇最原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
print(stem_1)

stem_2 = [stemmer.stem(t) for t in tokens_2]
print(stem_2)

# 初始化词性标注器
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print(pos_tag_1)

pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print(pos_tag_2)
