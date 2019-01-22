#!/usr/bin/env python3
# @Time    : 2018/10/7 11:09 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : word2vec_example.py
# @Software: PyCharm
# @Description 用20类新闻文本进行词向量训练

# 导入nltk和re工具包
import nltk
import re
# 从bs4导入BeautifulSoup
from bs4 import BeautifulSoup
from gensim.models import word2vec
# 从sklearn.datasets中导入20类新闻文本抓取器。
from sklearn.datasets import fetch_20newsgroups

# 使用新闻抓取器从互联网上下载所有数据，并且存储在变量news中。
news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target


# 定义一个函数名为news_to_sentences将每条新闻中的句子逐一剥离出来，并返回一个句子的列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news, features='lxml').get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)

    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())

    return sentences


sentences = []
for x in X:
    sentences += news_to_sentences(x)

print(len(sentences))

# Set values for various parameters
# 配置词向量的维度
num_features = 300    # Word vector dimensionality
# 保证被考虑的词汇的频度
min_word_count = 20   # Minimum word count
# 设定并行化训练使用CPU计算核心的数量，多核可用
num_workers = 2     # Number of threads to run in parallel
# 定义训练词向量的上下文窗口大小
context = 5        # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(sentences, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

print(model.wv.most_similar('morning'))
print(model.wv.most_similar('email'))



