#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: numpredict.py
@time: 2017/3/1 上午10:38
"""

import math
from random import random, randint

import optimization

def wine_price(rating, age):
    peak_age = rating - 50

    # Calculate price based on rating
    price = rating / 2
    if age > peak_age:
        # Past its peak, goes bad in 5 years
        price *= 5 - (age - peak_age)
    else:
        # Increases to 5x original value as it
        # approaches its peak
        price *= 5 * ((age + 1) / peak_age)
    if price < 0:
        price = 0
    return price


def wine_set1():
    rows = []
    for i in range(300):
        # Create a random age and rating
        rating = random() * 50 + 50
        age = random() * 50

        # Get reference price
        price = wine_price(rating, age)

        # Add some noise
        price*=(random()*0.4+0.8)

        # Add to the dataset
        rows.append({'input': (rating, age),
                     'result': price})
    return rows


def wine_set2():
    rows = []
    for i in range(300):
        # Create a random age and rating
        rating = random() * 50 + 50
        age = random() * 50
        aisle = float(randint(1, 20))
        bottle_size = [375.0, 750.0, 1500.0, 3000.0][randint(0, 3)]

        # Get reference price
        price = wine_price(rating, age)
        price *= (bottle_size / 750)
        price *= (random()*0.9 + 0.2)

        # Add some noise price*=(random( )*0.4+0.8)
        # Add to the dataset
        rows.append({'input': (rating, age, aisle, bottle_size),
                     'result': price})
    return rows


def euclidean(v1, v2):
    """
    Calculate the distance between the vector v1 and vector v2
    :param v1: vector v1
    :param v2: vector v2
    :return: the distance between the vector v1 and vector v2
    """
    d = 0.0
    for i in range(len(v1)):
        d += (v1[i] - v2[i]) ** 2

    return math.sqrt(d)


def get_distances(data, vec1):
    """

    :param data:
    :param vec1:
    :return:
    """
    distance_list = []
    for i in range(len(data)):
        vec2 = data[i]['input']
        distance_list.append((euclidean(vec1, vec2), i))
    distance_list.sort()
    return distance_list


def knn_estimate(data, vec1, k=3):
    # Get sorted distances
    dlist = get_distances(data, vec1)
    avg = 0.0

    # Take the average of the top k results
    for i in range(k):
        idx = dlist[i][1]
        avg += data[idx]['result']

    avg /= k
    return avg


def inverse_weight(dist, num=1.0, const=1.0):
    """
    User inverse function to convert distance to weight
    :param dist:
    :param num:
    :param const:
    :return:
    """
    return num/(dist + const)


def subtract_weight(dist, const=1.0):
    if dist > const:
        return 0
    else:
        return const - dist


def gaussian(dist, sigma=10.0):
    """
    Use gaussian function to convert distance to weight
    :param dist:
    :param sigma:
    :return:
    """
    return math.e**(-dist**2/(2*sigma**2))


def weighted_knn(data, vec1, k=5, weight_func=gaussian):
    # Get distances
    dlist = get_distances(data, vec1)
    avg = 0.0
    total_weight = 0.0

    # Get weighted average
    for i in range(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weight_func(dist)
        avg += weight * data[idx]['result']
        total_weight += weight

    avg /= total_weight
    return avg


def divide_data(data, test=0.05):
    train_set = []
    test_set = []
    for row in data:
        if random() < test:
            test_set.append(row)
        else:
            train_set.append(row)

    return train_set, test_set


def test_algorithm(algf, train_set, test_set):
    error = 0.0
    for row in test_set:
        guess = algf(train_set, row['input'])
        error += (row['result'] - guess)**2

    return error / len(test_set)


def cross_validate(algf, data, trials=100, test=0.05):
    error=0.0
    for i in range(trials):
        train_set, test_set = divide_data(data, test)
        error += test_algorithm(algf, train_set, test_set)

    return error / trials


def rescale(data, scale):
    scaled_data = []
    for row in data:
        scaled = [scale[i] * row['input'][i] for i in range(len(scale))]
        scaled_data.append({'input':scaled, 'result':row['result']})

    return scaled_data


def create_cost_function(algf, data):
    def costf(scale):
        sdata = rescale(data, scale)
        return cross_validate(algf, sdata, trials=10)

    return costf

if __name__ == "__main__":
    print wine_price(95.0, 3.0)
    print wine_price(95.0, 8.0)
    print wine_price(99.0, 1.0)
    print wine_price(99.0, 5.0)
    test_data = wine_set2()
    print test_data[0]
    print test_data[1]

    #print euclidean(test_data[0]["input"], test_data[1]["input"])

    print "**" * 20 + "knn estimate result"
    print knn_estimate(test_data, (99.0, 5.0))
    print knn_estimate(test_data, (99.0, 5.0), k=1)
    print knn_estimate(test_data, (99.0, 5.0), k=5)
    print knn_estimate(test_data, (99.0, 5.0), k=10)

    print "**" * 20 + "weighted knn"
    print weighted_knn(test_data, (99.0, 5.0))

    print "**" * 20 + "cross_validate"
    print cross_validate(knn_estimate, test_data)

    def knn3(d, v):
        return knn_estimate(d, v, 3)

    print cross_validate(knn3, test_data)

    def knn1(d, v):
        return knn_estimate(d, v, 1)

    print cross_validate(knn1, test_data)

    print cross_validate(weighted_knn, test_data)

    print "**" * 20 + "scaled_data"
    scaled_data = rescale(test_data, [10, 10, 0, 0.5])
    print scaled_data[0]
    print scaled_data[1]
    print cross_validate(knn3, scaled_data)
    print cross_validate(weighted_knn, scaled_data)

    weight_domain = [(0, 20)] * 4
    costf = create_cost_function(knn_estimate, test_data)
    print optimization.annealing_optimize(weight_domain, costf, step=2)

