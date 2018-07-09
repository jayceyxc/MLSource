#!/usr/local/bin/python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: text_utility.py
@time: 2017/8/8 09:04
"""

import os
import glob
import xlrd
import re

from jieba import analyse
from scipy.sparse import csr_matrix

valid_pattern = re.compile(u'[\u0020-\u007e\u0061-\u007a\u4e00-\u9fa5]+')


def purify_search_word(word):
    # word = word.decode('utf-8', 'ignore')
    return ' '.join(re.findall(valid_pattern, word))


def my_extract(documents, vocabulary_dict, topK=10):
    """
    analysis the documents, extract the topK words and its weight from each document
    fill the vocabulary_dict with the word and its index in the vocabulary dict, 
    return a csr_matrix represent the documents
    
    :param documents: the list of the contents, each element represents a document.
    :param vocabulary_dict: the word and its index in the vocabulary dict
    :return: a csr_matrix represent the documents
    """
    indptr = [0]
    indices = []
    data = []
    for document in documents:
        tfidf_vec = analyse.extract_tags(sentence=document, topK=topK, withWeight=True)
        # print("new document:")
        for i in range(len(tfidf_vec)):
            word = tfidf_vec[i][0]
            tfidf_value = tfidf_vec[i][1]
            # print(u"\t".join([tfidf_vec[i][0], str(tfidf_vec[i][1])]))
            index = vocabulary_dict.setdefault(word, len(vocabulary_dict))
            indices.append(index)
            data.append(tfidf_value)
        indptr.append(len(indices))
    # print data
    # print indices
    # print indptr
    return csr_matrix((data, indices, indptr), dtype=float)

EXCEL_PREFIX = "*.xlsx"


def get_documents(current_path=None, pattern=EXCEL_PREFIX):
    """
    analysis the file in the current_path, file name is filtered by pattern,
    and return the url and url category dict, url and url content dict.
    
    :param current_path: the path to analysis file
    :param pattern: the pattern to filter the file name
    :return: url and url category dict, url and url content dict
    """
    if current_path is None:
        current_path = os.path.split(os.path.realpath(__file__))[0]
    current_path = current_path.rstrip(os.sep)

    url_content = dict()
    url_cat = dict()
    for file_name in glob.glob(current_path + os.sep + pattern):
        excel_file = xlrd.open_workbook(file_name)
        for sheet_name in excel_file.sheet_names():
            sheet = excel_file.sheet_by_name(sheet_name)
            for i in range(sheet.nrows):
                if i == 0:
                    continue
                cat = None
                url = None
                title = ""
                keywords = ""
                description = ""
                content = ""
                try:
                    cat = sheet.cell(colx=0, rowx=i).value
                    # cat = str(cat).decode('utf-8', 'ignore')
                    cat = purify_search_word(cat)
                    cat = cat.strip()
                    if not cat:
                        continue
                except IndexError:
                    continue

                try:
                    url = sheet.cell(colx=1, rowx=i).value
                    # url = str(url).decode('utf-8', 'ignore')
                    url = url.strip()
                    if not url:
                        continue
                except IndexError:
                    continue

                # print url
                try:
                    title = sheet.cell(colx=2, rowx=i).value
                    # title = str(title).decode('utf-8', 'ignore')
                    title = purify_search_word(title)
                    title = title.strip()
                    if not title:
                        continue
                except IndexError:
                    pass

                try:
                    keywords = sheet.cell(colx=3, rowx=i).value
                    if keywords is not None:
                        # keywords = str(keywords).decode('utf-8', 'ignore')
                        keywords = purify_search_word(keywords)
                        keywords = keywords.strip()
                except IndexError:
                    pass

                try:
                    description = sheet.cell(colx=4, rowx=i).value
                    if description is not None:
                        # description = str(description).decode('utf-8', 'ignore')
                        description = purify_search_word(description)
                        description = description.strip()
                except IndexError:
                    pass

                try:
                    a_content = sheet.cell(colx=5, rowx=i).value
                    # a_content = str(a_content).decode('utf-8', 'ignore')
                    a_content = purify_search_word(a_content)
                    a_content = a_content.strip()
                    p_content = sheet.cell(colx=6, rowx=i).value
                    # p_content = str(p_content).decode('utf-8', 'ignore')
                    p_content = purify_search_word(p_content)
                    p_content = p_content.strip()
                    content = a_content + p_content
                    content = content.replace('\01', ' ')
                except IndexError:
                    pass

                if cat is None or url is None:
                    continue

                url_cat[url] = cat
                url_content[url] = " ".join([title, keywords, description, content])

    return url_cat, url_content


def get_content_tfidf(content, vocabulary, topK=10):
    """
    Analysis the content and return a csr_matrix of the top K words of the content and its tfidf weight
    
    :param content: the content to be analysis
    :param vocabulary: the vocabulary of the analysis, it is return by the classification train process
    :return: the csr_matrix that contains the top K words of the contents and its tfidf weight
    """
    indptr = [0]
    indices = []
    data = []
    tfidf_vec = analyse.extract_tags(sentence=content, topK=topK, withWeight=True)
    for i in range(len(tfidf_vec)):
        word = tfidf_vec[i][0]
        tfidf_value = tfidf_vec[i][1]
        # print(u"\t".join([tfidf_vec[i][0], str(tfidf_vec[i][1])]))
        if word in vocabulary:
            # print word, 'has tfidf', tfidf_value
            index = vocabulary.get(word)
            indices.append(index)
            data.append(tfidf_value)
    indptr.append(len(indices))
    # 【273， 872】
    # print(matrix.shape)
    # print "data", data
    # print data[indptr[0]:indptr[1]]
    # print "indices", indices
    # print indices[indptr[0]:indptr[1]]
    # print "indptr", indptr
    content_tfidf = csr_matrix((data, indices, indptr), shape=(1, len(vocabulary)), dtype=float)

    return content_tfidf
