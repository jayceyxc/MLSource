#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: url_clustering.py
@time: 2017/8/9 16:41
"""

from __future__ import print_function
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

from utility import text_utility


def main():
    train_url_cat, train_url_content = text_utility.get_documents(current_path="data", pattern="train_*.xlsx")
    vocabulary = dict()
    matrix = text_utility.my_extract(train_url_content.values(), vocabulary)
    labels = train_url_cat.values()
    true_k = len(set(train_url_cat.values()))
    print(true_k)
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=True)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(matrix)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(matrix, km.labels_, sample_size=1000))

    print()

    print("Top terms per cluster:")


    for clz in km.labels_:
        print(clz)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(true_k):
        print("Cluster index {0}: , name: {1}".format(i, km.labels_[i]), end=' ')
        for ind in order_centroids[i, :10]:
            print('{0} : {1}'.format(ind, vocabulary.keys()[ind]), end=' ')
        print()


if __name__ == "__main__":
    main()