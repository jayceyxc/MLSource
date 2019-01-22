#!/usr/bin/env python3
# @Time    : 2018/10/8 9:47 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : imdb_word2vec_gbc.py
# @Software: PyCharm
# @Description 使用GradientBoostingClassifier来进行影评情感分析

from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk.data
from nltk.corpus import stopwords
# from gensim.models import word2vec
from gensim.models import Word2Vec
# 从sklearn.ensemble导入GradientBoostingClassifier模型进行影评情感分析
from sklearn.ensemble import GradientBoostingClassifier
# 导入GridSearchCV用于超参数组合的网格搜索
from sklearn.model_selection import GridSearchCV


# 定义review_to_text函数，完成对原始评论的三项数据预处理任务
def review_to_text(review, remove_stopwords):
    # 任务一：去掉html标记
    raw_text = BeautifulSoup(review, features='html').get_text()
    # 任务二：去掉非字母字符
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    # 任务三：如果remove_words被激活，则进一步去掉评论中的停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    # 返回每条评论经此三项预处理任务的词汇列表
    return words


# 定义函数review_to_sentences逐条对影评进行分句
def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))

    return sentences


# 定义一个函数使用词向量产生文本特征向量
def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features, ), dtype='float32')
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word])

    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


# 定义另一个每条影评转化为基于词向量的特征向量（平均词向量）。
def get_avg_features_vecs(reviews, model, num_features):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype='float32')

    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter += 1

    return review_feature_vecs


# 从本地读入训练和测试数据集
train = pd.read_csv('../../Datasets/IMDB/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../../Datasets/IMDB/testData.tsv', delimiter='\t')
# 分别对原始训练和测试数据集上进行上述三项预处理
X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))

X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']


# 从本地读入未标记数据
unlabeled_train = pd.read_csv('../../Datasets/IMDB/unlabeledTrainData.tsv', delimiter='\t', quoting=3)

# 准备使用nltk的tokenizer对影评中的英文句子进行分割
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

corpora = []
# 准备用于训练词向量的数据
for review in unlabeled_train['review']:
    corpora += review_to_sentences(review, tokenizer)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# 开始词向量模型的训练
model = Word2Vec(corpora, workers=num_workers, size=num_features,
                 min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)

model_name = '../../Datasets/IMDB/300features_20minwords_10context',
# 可以将词向量模型的训练结果长期保存于本地硬盘
model.save(model_name)

# 直接读入已经训练好的词向量模型
model = Word2Vec.load('../../Datasets/IMDB/300features_20minwords_10context')
# 探查一下该词向量模型的训练成果
model.most_similar("man")

# 准备新的基于词向量表示的训练和测试特征向量
clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

train_data_vecs = get_avg_features_vecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(review_to_text(review, remove_stopwords=True))

test_data_vecs = get_avg_features_vecs(clean_test_reviews, model, num_features)

gbc = GradientBoostingClassifier()

# 配置超参数的搜索组合
params_gbc = {'n_estimators':[10, 100, 500], 'learning_rate':[0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}
gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)
gs.fit(train_data_vecs, y_train)

print(gs.best_score_)
print(gs.best_params_)

# 使用超参数调优之后的梯度上升树模型进行预测
result = gs.predict(test_data_vecs)
output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
output.to_csv('../../Datasets/IMDB/submission_w2v.csv', index=False, quoting=3)



