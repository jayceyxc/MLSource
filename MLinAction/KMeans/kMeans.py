#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: kMeans.py
@time: 29/11/2017 08:47
"""

import numpy as np


def load_data_set( filename ):
    """
    从文件读取数据到内存
    :param filename:
    :return:
    """
    data_mat = []
    with open(filename, mode='r') as fd:
        for line in fd:
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)

    return data_mat


def dist_euclid(vec_a, vec_b):
    """
    计算两个向量的欧几里得距离
    :param vec_a:
    :param vec_b:
    :return:
    """
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def rand_center(data_set, k):
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(data_set[:, j])
        range_j = float(np.max(data_set[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)

    return centroids


def kMeans(data_set, k, dist_meas=dist_euclid, create_cent = rand_center):
    m = np.shape(data_set)[0]
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroids = create_cent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(centroids[j, :], data_set[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist**2

        print(centroids)
        for cent in range(k):
            pts_in_cluster = data_set[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)

    return centroids, cluster_assment


def biKmeans(data_set, k, dist_meas=dist_euclid):
    m = np.shape(data_set)[0]
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(data_set, axis=0).tolist()[0]
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(np.mat(centroid0), data_set[j, :])**2

    while len(cent_list) < k:
        lowest_sse = np.inf
        for i in range(len(cent_list)):
            pts_in_current_cluster = data_set[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_cluster_ass = kMeans(pts_in_current_cluster, 2, dist_meas)
            sse_split = np.sum(split_cluster_ass[:, 1])
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print("sse_split, and not split", sse_split, sse_not_split)
            if (sse_split + sse_not_split) < lowest_sse:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_cluster_ass = split_cluster_ass.copy()
                lowest_sse = sse_split + sse_not_split

        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        print('the best_cent_to_split is: ', best_cent_to_split)
        print('the len of best cluster ass is: ', len(best_cluster_ass))

        cent_list[best_cent_to_split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :])
        cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_cluster_ass
        # print(cent_list)

    return cent_list, cluster_assment


if __name__ == '__main__':
    data_mat = np.mat(load_data_set('testSet.txt'))