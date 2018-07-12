#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: docclass_sqlite.py
@time: 2017/3/30 下午4:20
"""

from sqlite3 import dbapi2 as sqlite
import re
import math
from jieba import analyse
import jieba
import glob
import os
import xlrd
import excel_to_json
import sys
import traceback
import argparse
import time

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

chinese = re.compile(u'[\u4e00-\u9fa5]+')


def get_chinese_words(doc):
    # words = jieba.cut(doc)
    # words = analyse.extract_tags(doc, topK=20, withWeight=False)
    # return dict([(w, 1) for w in words])
    return analyse.extract_tags(doc, topK=10, withWeight=True)


class Classifier:
    WEIGHT_TITLE = 1.0
    WEIGHT_KEYWORDS = 0.8
    WEIGHT_DESCRIPTION = 0.8
    WEIGHT_CONTENT = 0.6

    KEY_TITLE = "title"
    KEY_KEYWORDS = "keywords"
    KEY_DESCRIPTION = "description"
    KEY_CONTENT = "content"

    UNKNOWN_CATEGORY = u"未知"

    def __init__(self, get_features, file_name=None):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        self.get_features = get_features
        self.weight_dict = {Classifier.KEY_TITLE: Classifier.WEIGHT_TITLE,
                            Classifier.KEY_KEYWORDS: Classifier.WEIGHT_KEYWORDS,
                            Classifier.KEY_DESCRIPTION: Classifier.WEIGHT_DESCRIPTION,
                            Classifier.KEY_CONTENT: Classifier.WEIGHT_CONTENT}

    def set_db_file(self, db_file_name):
        if os.path.exists(db_file_name):
            print("delete file {0}".format(db_file_name))
            os.remove(db_file_name)
        self.con = sqlite.connect(db_file_name)
        self.con.execute('create table if not exists fc(feature,category,weight)')
        self.con.execute('create table if not exists cc(category,count)')

    def inc_feature_weight(self, f, cat, weight=1.0):
        exist_weight = self.get_feature_weight(f, cat)
        if exist_weight == 0.0:
            self.con.execute("insert into fc values ('%s','%s', %f)"
                             % (f, cat, weight))
        else:
            self.con.execute(
                "update fc set weight=%f where feature='%s' and category='%s'"
                % (exist_weight + weight, f, cat))

    def get_feature_weight( self, f, cat):
        res = self.con.execute(
            'select weight from fc where feature="%s" and category="%s"'
            % (f, cat)).fetchone()
        if res is None:
            return 0
        else:
            return float(res[0])

    def inc_category_count(self, cat):
        count = self.get_category_count(cat)
        if count == 0:
            self.con.execute("insert into cc values ('%s',1)" % (cat))
        else:
            self.con.execute("update cc set count=%d where category='%s'"
                             % (count + 1, cat))

    def get_category_count(self, cat):
        res = self.con.execute('select count from cc where category="%s"'
                               % (cat)).fetchone()
        if res is None:
            # print("category {0} query res is None".format(cat))
            return 0
        else:
            # print res[0]
            return int(res[0])

    def get_all_categories(self):
        cur = self.con.execute('select category from cc')
        return [d[0] for d in cur]

    def total_count(self):
        # return the total count of all the train document
        res = self.con.execute('select sum(count) from cc').fetchone()
        if res is None:
            return 0
        return res[0]

    def train(self, item, cat):
        """

        :param item: the item should be a dict, key is doc_content, value is the weight of the weight
        :param cat: the category of the item
        :return: 
        """
        for key in item.keys():
            content = item[key]
            weight = 1.0
            if content is None or len(content) == 0:
                continue
            weight = self.weight_dict[key]
            features = self.get_features(content)
            # Increment the count for every feature with this category
            for f in features:
                self.inc_feature_weight(f[0], cat, weight * f[1])

        # Increment the count for this category
        self.inc_category_count(cat)
        self.con.commit()

    def feature_prob(self, feature, cat):
        if self.get_category_count(cat) == 0:
            return 0

        # The total number of times this feature appeared in this
        # category divided by the total number of items in this category
        return self.get_feature_weight(feature, cat) / self.get_category_count(cat)

    def weighted_prob(self, feature, cat, prf, weight=1.0, ap=0.5):
        # Calculate current probability
        basic_prob = prf(feature, cat)

        # Count the number of times this feature has appeared in all categories
        totals = sum([self.get_feature_weight(feature, c) for c in self.get_all_categories()])

        # Calculate the weighted average
        bp = ((weight * ap) + (totals * basic_prob)) / (weight + totals)
        return bp


class NaiveBayes(Classifier):
    """
    formula:
    Pr(Category|Document) = Pr(Document|Category) * Pr(Category) / Pr(Document)
    """

    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.thresholds = {}

    def doc_prob(self, item, cat):
        p = 1
        for key in item.keys():
            content = item[key]
            weight = 1.0
            if content is None or len(content) == 0:
                continue
            weight = self.weight_dict[key]
            features = self.get_features(content)
            # Multiply the probabilities of all the features together

            for f in features:
                p *= self.weighted_prob(f[0], cat, self.feature_prob, weight * f[1])
        return p

    def bayes_prob(self, item, cat):
        # calculate the Pr(Category), means the probability of the category
        # Because self.get_category_count() and self.total_count() return integer value, if don't convert to float
        # the divide result will zero.
        cat_prob = float(self.get_category_count(cat)) / self.total_count()

        # calculate the Pr(Document|Category), means for the given category,
        # the probability of the document belong to this category
        doc_prob = self.doc_prob(item, cat)

        # we have ignore the Pr(Document), because for the given document,
        # this value is same
        # print("category: {0}, document probability: {1}, category probability: {2}".format(cat, doc_prob, cat_prob))
        return doc_prob * cat_prob

    def set_threshold(self, cat, t):
        self.thresholds[cat] = t

    def get_threshold(self, cat):
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    def classify(self, item, default=None):
        prob_dict = {}
        # Find the category with the highest probability
        max_prob = 0.0
        best = None
        for cat in self.get_all_categories():
            prob_dict[cat] = self.bayes_prob(item, cat)
            if prob_dict[cat] > max_prob:
                max_prob = prob_dict[cat]
                best = cat

        if best is None:
            return default, 0.0

        # Make sure the probability exceeds threshold*next best
        for cat in prob_dict:
            if cat == best:
                continue
            if prob_dict[cat] * self.get_threshold(best) > prob_dict[best]:
                return default, 0.0
        return best, max_prob


class FisherClassifier(Classifier):
    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.minimums = {}

    def set_minimum(self, cat, min):
        self.minimums[cat] = min

    def get_minimum(self, cat):
        if cat not in self.minimums: return 0
        return self.minimums[cat]

    def cprob(self, f, cat):
        # The frequency of this feature in this category
        clf = self.feature_prob(f, cat)
        if clf == 0: return 0

        # The frequency of this feature in all the categories
        freqsum = sum([self.feature_prob(f, c) for c in self.get_all_categories()])

        # The probability is the frequency in this category divided by
        # the overall frequency
        p = clf / (freqsum)

        return p

    def fisher_prob(self, item, cat):
        # Multiply all the probabilities together
        p = 1
        for key in item.keys():
            content = item[key]
            weight = 1.0
            if content is None or len(content) == 0:
                continue
            weight = self.weight_dict[key]
            features = self.get_features(content)
            for f in features:
                p *= (self.weighted_prob(f[0], cat, self.cprob, weight * f[1]))

        # Take the natural log and multiply by -2
        fisher_score = -2 * math.log(p)

        # Use the inverse chi2 function to get a probability
        return self.invchi2(fisher_score, len(features) * 2)

    def invchi2(self, chi, df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1, df // 2):
            term *= m / i
            sum += term
        return min(sum, 1.0)

    def classify(self, item, default=None):
        # Loop through looking for the best result
        best = default
        max = 0.0
        for c in self.get_all_categories():
            p = self.fisher_prob(item, c)
            # Make sure it exceeds its minimum
            if p > self.get_minimum(c) and p > max:
                best = c
                max = p
        return best, max

#current_path = os.path.split(os.path.realpath(__file__))[0] + '/'
#TRAIN_DIR = current_path + 'trainfile/'
#TEST_DIR = current_path + 'testfile/'
EXCEL_PREFIX = "*.xlsx"
TRAIN_EXCEL_PATTERN = "train_*.xlsx"
TEST_EXCEL_PATTERN = "test_*.xlsx"


def chinese_train(cl, dir_name):
    current_path = os.path.split(os.path.realpath(__file__))[0]
    for file in glob.glob(current_path + os.sep + dir_name + os.sep + TRAIN_EXCEL_PATTERN):
        print(file)
        excel_file = xlrd.open_workbook(file)
        for sheet_name in excel_file._sheet_names:
            sheet = excel_file.sheet_by_name(sheet_name)
            for i in range(sheet.nrows):
                if i == 0:
                    continue
                cat = None
                url = None
                title = None
                keywords = None
                description = None
                content = None
                try:
                    cat = sheet.cell(colx=0, rowx=i).value
                    cat = str(cat).decode('utf-8', 'ignore')
                    cat = excel_to_json.purify_search_word(cat)
                    cat = cat.strip()
                    if not cat:
                        continue
                except IndexError:
                    continue

                try:
                    url = sheet.cell(colx=1, rowx=i).value
                    url = str(url).decode('utf-8', 'ignore')
                    url = url.strip()
                    if not url:
                        continue
                except IndexError:
                    continue

                # print(url)
                try:
                    title = sheet.cell(colx=2, rowx=i).value
                    title = str(title).decode('utf-8', 'ignore')
                    title = excel_to_json.purify_search_word(title)
                    title = title.strip()
                    if not title:
                        continue
                except IndexError:
                    continue

                try:
                    keywords = sheet.cell(colx=3, rowx=i).value
                    if keywords is not None:
                        keywords = str(keywords).decode('utf-8', 'ignore')
                        keywords = excel_to_json.purify_search_word(keywords)
                        keywords = keywords.strip()
                except IndexError:
                    continue

                try:
                    description = sheet.cell(colx=4, rowx=i).value
                    if description is not None:
                        description = str(description).decode('utf-8', 'ignore')
                        description = excel_to_json.purify_search_word(description)
                        description = description.strip()
                except IndexError:
                    continue

                try:
                    a_content = sheet.cell(colx=5, rowx=i).value
                    a_content = str(a_content).decode('utf-8', 'ignore')
                    a_content = excel_to_json.purify_search_word(a_content)
                    a_content = a_content.strip()
                    p_content = sheet.cell(colx=6, rowx=i).value
                    p_content = str(p_content).decode('utf-8', 'ignore')
                    p_content = excel_to_json.purify_search_word(p_content)
                    p_content = p_content.strip()
                    content = a_content + p_content
                    content = content.replace('\01', ' ')
                except IndexError:
                    continue

                if cat is None or url is None:
                    continue

                item = dict()
                item[Classifier.KEY_TITLE] = title
                if keywords is not None:
                    item[Classifier.KEY_KEYWORDS] = keywords

                if description is not None:
                    item[Classifier.KEY_DESCRIPTION] = description

                if content is not None:
                    item[Classifier.KEY_CONTENT] = content

                cl.train(item, cat)


'''
if __name__ == "__main__":
    cl = fisherclassifier(getchinesewords);
    cl.setdb("docfilter");
    sampletrain(cl)
    cl.classify('quick rabbit')
    cl.classify('quick money')
    cl.setminimum('bad', 0.8)
    cl.classify('quick money')
    cl.setminimum('good', 0.4)
    cl.classify('quick money')
'''

analyse.set_stop_words('jieba/country_city_stop_words.txt')
# analyse.set_stop_words('jieba/my_stop_words.txt')
jieba.load_userdict("jieba/extra_dict.txt")

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="category the host.")
    parser.add_argument('-c', '--classifier', choices=['fisher', 'bayes'], required=True, help='The classifier to use.')
    subparsers = parser.add_subparsers(help='additional help message',
                                       title='sub commands',
                                       description='valid sub commands',
                                       dest='sub_parser_name')
    parser_prepare = subparsers.add_parser("train", help="train the classifier")
    parser_prepare.add_argument("-d", "--data", required=True, help="the directory of the train data")
    parser_prepare = subparsers.add_parser("test", help="test the classifier")
    parser_prepare.add_argument("-d", "--data", required=True, help="the data file used to test the classifier")
    args = parser.parse_args()
    print(args)
    classifier = args.classifier
    cl = None
    if classifier == 'fisher':
        cl = FisherClassifier(get_chinese_words)
    elif classifier == 'bayes':
        cl = NaiveBayes(get_chinese_words)
    else:
        assert("wrong classifier parameter")
        sys.exit(0)
    cl.set_db_file("docfilter")
    sub_command = args.sub_parser_name
    if sub_command == "train":
        dir_name = args.data
        chinese_train(cl, dir_name)
    elif sub_command == "test":
        filename = args.data
        full_filename = os.path.split(os.path.realpath(__file__))[0] + os.path.sep + filename
        with open(full_filename, "r") as data_file:
            for line in data_file:
                try:
                    line = line.strip()
                    if line is None or len(line) == 0:
                        continue
                    url, title, keywords, description, a_content, p_content = line.split(",")
                    content = a_content + p_content
                    if not chinese.search(title.decode('utf8')) or not chinese.search(content.decode('utf8')):
                        print u"\t".join([Classifier.UNKNOWN_CATEGORY, "0.0", url])
                        continue
                    item = {Classifier.KEY_TITLE: title, Classifier.KEY_KEYWORDS: keywords,
                            Classifier.KEY_DESCRIPTION: description, Classifier.KEY_CONTENT: content}
                    cat, prob = cl.classify(item, default=Classifier.UNKNOWN_CATEGORY)
                    print u"\t".join([cat, str(prob), url])
                except UnicodeDecodeError:
                    traceback.print_exc()
                    pass
    """
    # clf = FisherClassifier(get_features=get_chinese_words)
    # clf.set_db_file("fisher_docfilter")
    clf = NaiveBayes(get_features=get_chinese_words)
    clf.set_db_file("naviebayes_docfilter")
    train_dir = "data"
    chinese_train(clf, train_dir)
    # time.sleep(10)
    current_path = os.path.split(os.path.realpath(__file__))[0]
    total_count = 0
    right_count = 0
    cat_true = list()
    cat_pred = list()
    for file_name in glob.glob(current_path + os.sep + "data" + os.sep + TEST_EXCEL_PATTERN):
        # print(file_name)
        excel_file = xlrd.open_workbook(file_name)
        for sheet_name in excel_file._sheet_names:
            sheet = excel_file.sheet_by_name(sheet_name)
            for i in range(sheet.nrows):
                cat = None
                url = None
                title = None
                keywords = None
                description = None
                content = None
                try:
                    cat = sheet.cell(colx=0, rowx=i).value
                    cat = str(cat).decode('utf-8', 'ignore')
                    cat = excel_to_json.purify_search_word(cat)
                    cat = cat.strip()
                    if not cat:
                        continue
                except IndexError:
                    continue

                try:
                    url = sheet.cell(colx=1, rowx=i).value
                    url = str(url).decode('utf-8', 'ignore')
                    url = url.strip()
                    if not url:
                        continue
                except IndexError:
                    continue

                # print(url)
                try:
                    title = sheet.cell(colx=2, rowx=i).value
                    title = str(title).decode('utf-8', 'ignore')
                    title = excel_to_json.purify_search_word(title)
                    title = title.strip()
                    if not title:
                        continue
                except IndexError:
                    continue

                try:
                    keywords = sheet.cell(colx=3, rowx=i).value
                    if keywords is not None:
                        keywords = str(keywords).decode('utf-8', 'ignore')
                        keywords = excel_to_json.purify_search_word(keywords)
                        keywords = keywords.strip()
                except IndexError:
                    continue

                try:
                    description = sheet.cell(colx=4, rowx=i).value
                    if description is not None:
                        description = str(description).decode('utf-8', 'ignore')
                        description = excel_to_json.purify_search_word(description)
                        description = description.strip()
                except IndexError:
                    continue

                try:
                    a_content = sheet.cell(colx=5, rowx=i).value
                    a_content = str(a_content).decode('utf-8', 'ignore')
                    a_content = excel_to_json.purify_search_word(a_content)
                    a_content = a_content.strip()
                    p_content = sheet.cell(colx=6, rowx=i).value
                    p_content = str(p_content).decode('utf-8', 'ignore')
                    p_content = excel_to_json.purify_search_word(p_content)
                    p_content = p_content.strip()
                    content = a_content + p_content
                    content = content.replace('\01', ' ')
                except IndexError:
                    continue

                if cat is None or url is None:
                    continue

                item = dict()
                item[Classifier.KEY_TITLE] = title
                if keywords is not None:
                    item[Classifier.KEY_KEYWORDS] = keywords

                if description is not None:
                    item[Classifier.KEY_DESCRIPTION] = description

                if content is not None:
                    item[Classifier.KEY_CONTENT] = content

                predict_cat, prob = clf.classify(item, default=Classifier.UNKNOWN_CATEGORY)
                cat_true.append(cat)
                cat_pred.append(predict_cat)
                total_count += 1
                if cat == predict_cat:
                    right_count += 1
                print("expected: {0}, predicted: {1}, probability: {2}, url: {3}".format(cat, predict_cat, prob, url))

    print("right count: {0}, total count: {1}, accuracy: {2}".format(right_count, total_count,
                                                                     float(right_count) / total_count))
    cat_true_array = np.array(cat_true)
    cat_pred_array = np.array(cat_pred)
    for average in ["micro", "macro", "weighted"]:
        precision, recall, fscore, support = precision_recall_fscore_support(cat_true_array, cat_pred_array, average=average)
        print("precision: {0}, recall: {1}, fscore: {2}, support: {3}".format(precision, recall, fscore, support))
