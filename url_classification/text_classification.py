#!/usr/bin/env python3
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: text_classification.py
@time: 2017/8/6 12:58
"""

import traceback
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utility import text_utility


def url_classification():
    url_cat, url_content = text_utility.get_documents(current_path="data")
    vocabulary = dict()
    matrix = text_utility.my_extract(url_content.values(), vocabulary)
    features = dict((value, key) for key, value in vocabulary.iteritems())
    clf = MultinomialNB().fit(matrix, url_cat.values())
    # content = u"""河源新闻·南方网·广东媒体融合第一平台·Heyuan News河源新闻,新闻,广东新闻,中国,广东,河源,槎城,古邑客家,经济,源城,,东源,和平,龙川,紫金,连平,高新区,江东新区,万绿湖,旅游美食,民生,社会,政治,法治,时事,质量,图库河源新闻网是集新闻信息、互动社区、行业资讯为一体的地方综合门户网站，为河源广大网民提供最全面、最快捷的本地化资讯，并致力于打造河源政经主流、民生关注网站，主打新闻、网络问政、民生热线三大品牌。- 或用以下帐号直接登录 - Heyuan NEWS 河源新闻[退出] EN × 极速注册 忘记密码？ QQ登录 微博登录 广州 深圳 珠海 汕头 佛山 韶关 河源 梅州 惠州 汕尾 东莞 中山 江门 阳江 湛江 茂名 肇庆 清远 潮州 揭阳 云浮 快报 要闻 头条 大图 政要 独家 创新 南融 产业 脱贫 影像 城乡 园区 交通 推荐 环保 更多 微话题 张文调研市驻穗办要求 做好招商引资工作 更多头条 要闻 政要 河源明年实现乡镇污水处理设施全覆盖 遥控航空模型飞行员执照培训考核落地河源 河源214位学生参加舞蹈考级 河源大学生创园征集作品评选结果出炉 河源将优先保障城乡残疾人基本住房 河源将在行政管理事项中使用信用记录和信用报告 河源将设立新兴产业发展引导基金 市中级人民法院罗婷芳获“全国优秀法官”称号 吴善平任河源市人大常委会党组副书记 市侨联召开六届二次全委会议 李茂辉当选主席 张文彭建文率队东源调研要求 强化产城融合发展 张文彭建文三县调研要求 以项目促发展惠民生 赖小卫任连平县人武部党委第一书记 彭建文昨到巴伐利亚庄园现场办公 谢耀琪、范秀燎任河源市副市长 张文寄语市检察院 为振兴发展提供有力司法保障 东源上莞"灯日" 寓意人丁兴旺香火传承 今年河源计划投658亿元用于基础设施建设 紫金甘田村逾180盏路灯全线亮灯 大广高速迎来春运返乡车流高峰 快报 受冷空气影响 本周河源气温低至7℃ 河源开设“青春情暖驿站”返乡人员享受免费服务 河源火车站开启春运模式 旅客日均达4000人次 创新 市中级人民法院罗婷芳获“全国优秀法官”称号 河源学习全国首部市场监管条例 省科技厅助力东源打造县域创新驱动发展品牌 产业 紫金力争2018年底茶园面积种植达3万亩 高新区2企业入选省两化融合贯标试点 高新区企业招聘会2500人达成就业意向 城乡 连平春节实现森林“零火情” 连平部署省、市重点项目推进工作 江东新区节日期间社会秩序稳定 独家 河源雅居乐工地仅地下车库发生局部坍塌 河源市东源县仙塘镇：文明鲜花绽放美丽乡村 河源3园区获评省优秀产业园 南融 深圳大学加强教育帮扶助力河源振兴发展 深河一体化驱动河源发展蜕变 河源致力打造深莞惠产业发展“大腹地” 脱贫 东源灯塔镇“一户一策”精准建档立册 截至目前深圳福田助力和平1903人实现脱贫 东源法院借力华南农业大学科技力量开展精准扶贫 园区 江东新区全力推进“两个起步区”建设 和平工业园企业新增三千个就业岗位 江东新区加快完善园区配套 力促项目动工建设 河源启动应急机制应对办证高峰 女子顺手牵羊偷手机 和平警方快速侦破 源城公安侦破一起盗窃自行车案件 市民可通过缴费机自助缴纳水费 河源家政市场火热 八成家庭请保姆带小孩 有人往万绿湖里倒建筑废料？官方回应：施工方准备做护坡 和平警方侦破一宗命案 男子情绪失控持刀伤人 河源儿童城预计6月试营业 交通 河惠莞高速东源段沿线四个标段开始施工 龙川加快推进河惠莞高速公路征地拆迁工作 东源召开专题会议部署河惠莞高速沿线镇征收工作 河源客运站码头等地将实时监控 环保 龙川依法强制关停14间非法砖厂 元宵节河源禁止辖区内销售和燃放“孔明灯” 河源将继续大力开展大气环境综合整治 河源住建局牵头制定《方案》整治工地扬尘 更多 舞台剧《赤子丰碑》河源上演 河源瓦溪服务区帮助被落下的旅客坐上大巴 河源“最美家庭”评选正式开始 男子恶意透支信用卡逾12万元被刑拘 影像 河源县区赏花攻略大全 广州粤羽精英走进消防警营参观体验 河源紫金花朝戏与粤剧同台竞艺 化龙路华丽变身，美得不像话！ 龙津古渡景观新面目 二月河源东源嶂下村千亩山楂开成花海 南方网 www.southcn.com 南方报业传媒集团简介 网站简介 网站地图 广告服务 诚聘英才 联系我们 法律声明 友情链接"""
    # content_matrix = get_content_tfidf(content, vocabulary)
    # clf.predict(content_matrix)
    return clf, vocabulary, features


def text_classification():
    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print(X_train_counts.shape)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    """
    Now that we have our features, we can train a classifier to try to predict the category of a post. 
    Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task.
    """
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

    docs_new = ["OpenGL on the GPU is fast", "USA military airplanes"]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))


def text_classification_pipeline_version():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
        # ('clf', GaussianNB())
    ])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
    docs_new = ["OpenGL on the GPU is fast", "USA military airplanes"]
    predicted = text_clf.predict(docs_new)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))

    """
    Evaluating the predictive accuracy of the model is equally easy
    """
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted_test = text_clf.predict(docs_test)
    print(np.mean(predicted_test == twenty_test.target))


def text_classification_use_svm():
    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss="hinge", penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
    ])
    text_clf.fit(twenty_train.data, twenty_train.target)

    """
    Evaluating the predictive accuracy of the model is equally easy
    """
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    print(np.mean(predicted==twenty_test.target))
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
    print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])
    print(gs_clf.best_score_)


if __name__ == "__main__":
    # text_classification()
    # text_classification_pipeline_version()
    # text_classification_use_svm()
    clf, vocabulary, features = url_classification()
    with open("test.txt", mode='r') as fd:
        first = True
        for line in fd:
            if first:
                first = False
                continue

            try:
                indptr = [0]
                indices = []
                data = []
                line = line.strip()
                line = line.decode('utf-8', 'ignore')
                segs = line.split(',')
                if len(segs) != 6:
                    continue
                url, title, keywords, desc, a_content, p_content = line.split(',')
                content = " ".join([title, keywords, desc, a_content, p_content])
                content_matrix = text_utility.get_content_tfidf(content, vocabulary)

                cat = clf.predict(content_matrix)

                print(u"\t".join([url, cat[0]]))
            except UnicodeDecodeError as ude:
                traceback.print_exc()
                continue
