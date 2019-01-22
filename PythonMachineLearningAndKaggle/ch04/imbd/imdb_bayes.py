#!/usr/bin/env python3
# @Time    : 2018/10/8 8:25 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : imdb_bayes.py
# @Software: PyCharm
# @Description IMDB影评得分估计竞赛编码 链接：https://www.kaggle.com/c/word2vec-nlp-tutorial

# 导入pandas用于读取和写入数据操作
import pandas as pd
# 从bs4导入BeautifulSoup用于整理原始文本
from bs4 import BeautifulSoup
# 导入正则表达式工具包
import re
# 从nltk.corpus里导入停用词列表
from nltk.corpus import stopwords
# 导入文本特性抽取器CountVectorizer与TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 从scikit-learn中导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 导入Pipeline用于方便搭建系统流程
from sklearn.pipeline import Pipeline
# 导入GridSearchCV用于超参数组合的网格搜索
from sklearn.model_selection import GridSearchCV

# 从本地读入训练和测试数据集
train = pd.read_csv('../../Datasets/IMDB/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../../Datasets/IMDB/testData.tsv', delimiter='\t')

# 查验一下前几条训练数据
print(train.head())

# 查验一下前几条测试数据
print(test.head())


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


# 分别对原始训练和测试数据集上进行上述三项预处理
X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))

X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']

# 使用Pipeline搭建两组使用朴素贝叶斯模型的分类器，区别在于分别使用CountVectorizer与TfidfVectorizer对文本特征进行抽取
pip_count = Pipeline(
    [
        ('count_vec', CountVectorizer(analyzer='word')),
        ('mnb', MultinomialNB())
    ]
)
pip_tfidf = Pipeline(
    [
        ('tfidf_vec', TfidfVectorizer(analyzer='word')),
        ('mnb', MultinomialNB())
    ]
)

# 分别配置用于模型超参数搜索的组合
params_count = {
    'count_vec__binary': [True, False],
    'count_vec__ngram_range': [(1, 1), (1, 2)],
    'mnb__alpha': [0.1, 1.0, 10.0]
}
params_tfidf = {
    'tfidf_vec__binary': [True, False],
    'tfidf_vec__ngram_range': [(1, 1), (1, 2)],
    'mnb__alpha': [0.1, 1.0, 10.0]
}

# 使用采用4折交叉验证的方法对使用CountVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
gs_count.fit(X_train, y_train)

# 输出交叉验证中最佳的准确性得分以及超参数组合
print(gs_count.best_score_)
print(gs_count.best_params_)

# 以最佳的超参数组合配置模型并对测试数据进行预测
count_y_predict = gs_count.predict(X_test)

# 使用采用4折交叉验证的方法对使用TfidfVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
gs_tfidf.fit(X_train, y_train)

# 输出交叉验证中最佳的准确性得分以及超参数组合
print(gs_tfidf.best_score_)
print(gs_tfidf.best_params_)

# 以最佳的超参数组合配置模型并对测试数据进行预测
tfidf_y_predict = gs_tfidf.predict(X_test)

# 使用pandas对需要提交的数据进行格式化
submission_count = pd.DataFrame(
    {
        'id': test['id'],
        'sentiment': count_y_predict
    }
)
submission_tfidf = pd.DataFrame(
    {
        'id': test['id'],
        'sentiment': tfidf_y_predict
    }
)

# 结果输出到本地硬盘
submission_count.to_csv('../../Datasets/IMDB/submission_count.csv', index=False)
submission_tfidf.to_csv('../../Datasets/IMDB/submission_tfidf.csv', index=False)









