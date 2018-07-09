#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: usertag.py
@time: 2017/7/19 17:34
"""

from pyspark import SparkConf, SparkContext, SparkFiles
from rule_manager import RuleParser
from config import Config

class fixLogFilter:

    def __init__(self, fieldIniPath):
        self.fieldIniPath = fieldIniPath

    def func(self, line):
        conf = Config(self.fieldIniPath)
        field_list = ['NUM', 'ADSL']
        NUM, ADSL = conf.get_field_indexes('map_fix', field_list)
        segs = [s.strip() for s in line.split('\01')]
        if len(segs) > NUM and segs[ADSL] is not None and len(segs[ADSL]) > 0:
            return True
        else:
            return False


if __name__ == "__main__":
    conf = SparkConf()
    conf.setAppName("UserTag")
    conf.setMaster("spark://192.168.3.110:7077")
    sc = SparkContext(conf=conf)

    fieldIniPath = SparkFiles.get("field.ini")
    rulePath = SparkFiles.get("rule0.csv")

    parser = RuleParser(net_type="fix")
    parser.parse_rules(rulePath)

    filePath = "hdfs://192.168.3.110:8020/user/yuxuecheng/dpc_log/20170319/mtdpc_visit_20170319*"
    outfilePath = "hdfs://192.168.3.110:8020/user/yuxuecheng/dpc_log/device_fimaly_out"
    originalRDD = sc.textFile(filePath)
    filteredRDD = originalRDD.filter(fixLogFilter())
