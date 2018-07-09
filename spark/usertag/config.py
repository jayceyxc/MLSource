#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 各mapreduce分步输入字段分区配置文件（field.ini）解析模块
"""

# 该模块不需在sys.path中加入当前目录，因为没有import非自带模块
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
# sys.path.append('.')
import ConfigParser

# 对配置文件field.ini进行解析
# 所得到的ConfigParser对象放置在一个单独模块中，可通过import进行接入控制
# 且避免重复解析配置


class Config:
    def __init__(self, configFile):
        self.config_parser = ConfigParser.ConfigParser()
        self.config_parser.read(configFile)

    def is_debug(self):
        """
        Whether debug mode in on
        :return: True if is in debug mode, False if not in debug mode
        """
        ret = int(self.config_parser.get("common", "DEBUG"))
        if ret == 0:
            return False
        else:
            return True


    def get_threshold(self, default=1):
        """
        Get the threshold value of the label appear times
        :param default: the default value of the thread value
        :return: 
        """
        try:
            threshold = int(self.config_parser.get("common", "THRESHOLD"))
            return threshold
        except KeyError:
            return default


    def get_max_label_number(self, default=1):
        """
        Get the max label numbers
        :param default: 
        :return: 
        """
        try:
            max_label_number = int(self.config_parser.get("common", "MAX_LABELS"))
            return max_label_number
        except KeyError:
            return default

    def get_field_indexes(self, section, field_names):
        """ 从config_parser中某一section寻找对应field_names列表的各个配置值
        :param section: 目标config_parser配置分区
        :param field_names: 需要的配置键（字符串）列表
        :return: 配置值列表，若为int或float则做相应转换
        """
        ret_list = []
        for fn in field_names:
            ret = None
            try:
                ret = int(self.config_parser.get(section, fn))
            except ValueError:
                try:
                    ret = float(self.config_parser.get(section, fn))
                except ValueError:
                    pass
            except:
                pass
            ret_list.append(ret)
        return ret_list

