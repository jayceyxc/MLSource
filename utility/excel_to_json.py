#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: excel_to_json.py
@time: 05/12/2017 11:39
"""

import xlrd
import xlwt
import json
import re
import sys


valid_pattern = re.compile(u'[\u0020-\u007e\u0061-\u007a\u4e00-\u9fa5]+')


def purify_search_word(word):
    word = word.decode('utf-8', 'ignore')
    return ' '.join(re.findall(valid_pattern, word))


class ExcelAdaptor(object):
    def __init__(self):
        pass

    @staticmethod
    def excel_to_json(excel_file_name):
        excel_file = xlrd.open_workbook(excel_file_name)
        book_d = {}
        for sheet_name in excel_file._sheet_names:
            sheet = excel_file.sheet_by_name(sheet_name)
            sheed_d = {}
            for i in range(sheet.nrows):
                try:
                    key = sheet.cell(colx=0, rowx=i).value
                    # key = str(key).decode('utf-8', 'ignore')
                    key = purify_search_word(key)
                    key = key.strip()
                    if not key:
                        continue
                except IndexError:
                    continue
                try:
                    value = sheet.cell(colx=1, rowx=i).value
                    # value = str(value).decode('utf-8', 'ignore')
                    value = purify_search_word(value)
                    value = value.strip()
                except IndexError:
                    value = None
                if key:
                    sheed_d[key] = value
            else:
                book_d[sheet_name.strip()] = sheed_d
        return book_d

    @staticmethod
    def json_to_excel(excel_file_name, json_obj):
        workbook = xlwt.Workbook()
        for key, value_dict in json_obj.iteritems():
            sheet = workbook.add_sheet(key)
            for index, (name, value) in enumerate(value_dict.iteritems()):
                sheet.write(r=index, c=0, label=name)
                if value:
                    sheet.write(r=index, c=1, label=value)
        workbook.save(excel_file_name)


if __name__ == '__main__':
    ss = json.dumps(ExcelAdaptor.excel_to_json(sys.argv[1]), ensure_ascii=False, indent=2)
    print(ss)
    # adaptor.json_to_excel(excel_file_name='test.xls', json_obj=json.loads(ss, encoding='utf-8'))