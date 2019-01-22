#!/usr/bin/env python3
# @Time    : 2018/10/6 9:02 AM
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : feature_extraction_test.py
# @Software: PyCharm
# @Description 特征提取的一些方法

from sklearn.feature_extraction import DictVectorizer


def dict_vectorizer_test():
    """
    DictVectorizer对使用字典存储的数据进行特征抽取与向量化
    :return:
    """
    # 定义一组字典列表，用来表示多个数据样本（每个字典代表一个数据样本）。
    measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.},
                    {'city': 'San Fransisco', 'temperature': 18.}]
    # 初始化DictVectorizer特征抽取器
    vec = DictVectorizer()
    results = vec.fit_transform(measurements)
    print(results)
    print(results.toarray())
    print(vec.get_feature_names())


if __name__ == '__main__':
    dict_vectorizer_test()




