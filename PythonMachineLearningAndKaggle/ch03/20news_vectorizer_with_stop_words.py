#!/usr/bin/env python3
# @Time    : 2018/10/6 1:24 PM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : 20news_vectorizer_with_stop_words.py
# @Software: PyCharm
# @Description 分别使用CountVectorizer与TfidfVectorizer，并且去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试

# 从sklearn.datasets里导入20类新闻文本数据抓取器。
from sklearn.datasets import fetch_20newsgroups
# 从sklearn.model_selection导入train_test_split模块用于分割数据集。
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text里导入CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# 从sklearn.naive_bayes里导入朴素贝叶斯分类器。
from sklearn.naive_bayes import MultinomialNB
# 从sklearn.metrics 导入 classification_report。
from sklearn.metrics import classification_report


# 从互联网上即时下载新闻样本,subset='all'参数代表下载全部近2万条文本存储在变量news中。
news = fetch_20newsgroups(subset='all')
# 对news中的数据data进行分割，25%的文本用作测试集；75%作为训练集。
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 分别使用停用词过滤配置初始化CountVectorizer与TfidfVectorizer。
tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')
# 只使用词频统计的方式将原始训练和测试文本转化为特征向量。
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

count_filter_vec = CountVectorizer(analyzer='word', stop_words='english')
# 只使用词频统计的方式将原始训练和测试文本转化为特征向量。
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

# 初始化默认配置的朴素贝叶斯分类器，并对CountVectorizer后的数据进行预测与准确性评估。
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer by filtering stopwords): {0}'
      .format(mnb_count_filter.score(X_count_filter_test, y_test)))
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)

# 初始化另一个默认配置的朴素贝叶斯分类器，并对TfidfVectorizer后的数据进行预测与准确性评估。
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes (TfidfVectorizer by filtering stopwords):{0}'
      .format(mnb_tfidf_filter.score(X_tfidf_filter_test, y_test)))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)

# 对上述两个模型进行更加详细的性能评估。
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))
print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))
