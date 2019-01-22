#!/usr/bin/env python3
# @Time    : 2018/10/7 9:29 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : grid_search_example.py
# @Software: PyCharm
# @Description 使用单线程对文本分类的朴素贝叶斯模型的超参数组合执行网格搜索。

# 从sklearn.datasets中导入20类新闻文本抓取器。
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
# 导入支持向量机（分类）模型。
from sklearn.svm import SVC
# 导入TfidfVectorizer文本抽取器。
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入Pipeline。
from sklearn.pipeline import Pipeline
# 从sklearn.grid_search中导入网格搜索模块GridSearchCV。
from sklearn.model_selection import GridSearchCV
import numpy as np


# 使用新闻抓取器从互联网上下载所有数据，并且存储在变量news中。
news = fetch_20newsgroups(subset='all')

# 对前3000条新闻文本进行数据分割，25%文本用于未来测试。
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25, random_state=33)

# 使用Pipeline 简化系统搭建流程，将文本抽取与分类器模型串联起来。
clf = Pipeline(
    [('vect', TfidfVectorizer(stop_words='english', analyzer='word')),
     ('svc', SVC())]
)

# 这里需要试验的2个超参数的的个数分别是4、3， svc__gamma的参数共有10^-2, 10^-1, 10^0, 10^1, svc__C的参数共有10^-1, 10^0, 10^1 。
# 这样我们一共有12种的超参数组合，12个不同参数下的模型。
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

# 将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV。请大家务必注意refit=True这样一个设定 。
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索。
_ = gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)

# 输出最佳模型在测试集上的准确性。
print(gs.score(X_test, y_test))

