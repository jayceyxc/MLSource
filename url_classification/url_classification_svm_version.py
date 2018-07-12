#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: url_classification_svm_version.py
@time: 2017/4/28 上午10:37
"""

# reload(sys)
# sys.setdefaultencoding('utf8')
import os

import jieba
import matplotlib.pyplot as plt
import numpy as np
from jieba import analyse
from sklearn import model_selection
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from utility import text_utility


def my_tokenizer( content ):
    return analyse.extract_tags(sentence=content, topK=5, withWeight=True)


def extract_chinese( documents ):
    tfidf_vec = TfidfVectorizer(tokenizer=my_tokenizer, use_idf=True, encoding='utf-8', analyzer="word")
    word_tfidf = tfidf_vec.fit_transform(documents)
    word_list = tfidf_vec.get_feature_names()
    for doc_index in range(word_tfidf.shape[0]):
        for word_index in word_tfidf[doc_index].indices:
            print("document ", doc_index, "has word", word_list[word_index], "with tfidf", word_tfidf[
                doc_index, word_index])


def train_param( X, y ):
    """
    :param X: array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    :param y: array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    :return: 
    """
    param_range = np.logspace(-6, -1)
    train_scores, test_scores = model_selection.validation_curve(svm.SVC(kernel="linear"), X, y, param_name='gamma',
                                                                 param_range=param_range, cv=10, scoring="accuracy",
                                                                 n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score", alpha=0.2,
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", alpha=0.2,
                 color="blue", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color="blue", lw=lw)
    plt.legend(loc="best")
    plt.show()


def grid_search_svc(X, y):
    """
    :param X: array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    :param y: array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    :return: 
    """

    # Split the dataset in two equal parts
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


if __name__ == "__main__":
    jieba.load_userdict("dict" + os.sep + "user.dict")
    analyse.set_stop_words(os.path.join("dict", "stop_words.txt"))
    text_1 = u"我爱北京天安门"
    text_2 = u"我爱吃西瓜，可以买西瓜吃么"
    text_3 = u"在前面的例子中，我们使用的方法是为每个单词创建一个特征。那么，将两个词一起作为一个特征会怎样呢？这正是二元语法" \
             u"(bigrams)所要考虑的。二元语法(或者更具一般性的n元语法)中，一个词是否出现，与它相邻的词语密切相关。当然，也可以" \
             u"混用一元语法(unigram)和n元语法，为文档创建更丰富的特征。通过一个简单的例子，让我们看看n元语法的工作原理"
    document_list = [text_1, text_2, text_3]
    # url_cat, url_content = text_utility.get_documents(current_path="data")
    url_cat, url_content = text_utility.get_documents(current_path="data", pattern="Query_train_content.xlsx")
    # print json.dumps(url_cat.values(), indent=2, ensure_ascii=False)
    # print json.dumps(url_content.values(), indent=2, ensure_ascii=False)
    # extract_chinese(document_list)
    vocabulary = dict()
    matrix = text_utility.my_extract(list(url_content.values()), vocabulary)
    features = dict((value, key) for key, value in vocabulary.items())
    # print matrix.toarray()
    # for i in range(len(matrix.indptr) - 1):
    #     for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]:
    #         print "document", i, "has word", features[j], "has tfidf value", matrix[i, j]

    # svc = svm.SVC(decision_function_shape="ovo")
    svc = svm.SVC(kernel="linear")
    # svc = svm.LinearSVC()
    svc.fit(matrix, list(url_cat.values()))
    # print type(svc.classes_)
    # print svc.classes_
    # print json.dumps(svc.classes_.tolist(), indent=2, ensure_ascii=False)
    # print(svc.n_support_)
    # print(svc.support_)
    # print(svc.support_vectors_)
    # print len(vocabulary)
    # print json.dumps(vocabulary, indent=2, ensure_ascii=False)
    # right_count = 0
    # total_count = 0
    # with open("data/test_ansi.csv", mode='r') as fd:
    #     first = True
    #     for line in fd:
    #         if first:
    #             first = False
    #             continue
    #
    #         try:
    #             indptr = [0]
    #             indices = []
    #             data = []
    #             line = line.strip()
    #             segs = line.split(',')
    #             if len(segs) != 7:
    #                 continue
    #             expected_cat, url, title, keywords, desc, a_content, p_content = line.split(',')
    #             content = " ".join([title, keywords, desc, a_content, p_content])
    #             # print "content: ", content
    #
    #             m = text_utility.get_content_tfidf(content, vocabulary)
    #
    #             # print m
    #             cat = svc.predict(m)
    #             total_count += 1
    #             if expected_cat == cat[0]:
    #                 right_count += 1
    #             # print type(cat)
    #             # print url
    #             # print cat
    #             # print target
    #             print(u"\t".join([url, cat[0]]))
    #         except UnicodeDecodeError as ude:
    #             traceback.print_exc()
    #             continue
    #
    # print("right count: {0}, total count: {1}, accuracy: {2}".format(right_count, total_count, float(right_count) / total_count))
    # print url_cat.values()
    # train_param(matrix, url_cat.values())
    grid_search_svc(matrix, list(url_cat.values()))
