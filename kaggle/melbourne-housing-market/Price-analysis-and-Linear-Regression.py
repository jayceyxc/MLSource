#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: Price-analysis-and-Linear-Regression.py
@time: 2017/5/5 上午9:18
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    dataframe = pd.read_csv("/Users/yuxuecheng/data/input/Melbourne_housing_extra_data.csv", parse_dates=[7],
                            dayfirst=True)
    len(dataframe["Date"].unique()) / 4
    # 选取Type值为h的行，并且按Date进行排序分组，最后计算每一列按日期分组后每一组数据的标准差，即每列数据每天都有一个标准差
    var = dataframe[dataframe["Type"] == "h"].sort_values("Date", ascending=False).groupby("Date").std()
    # 选取Type值为h的行，并且按Date进行排序分组，最后计算每一列按日期分组后每一组数据的数据条数，即每列数据每天都有一个数据条数
    count = dataframe[dataframe["Type"] == "h"].sort_values("Date", ascending=False).groupby("Date").count()
    # 选取Type值为h的行，并且按Date进行排序分组，最后计算每一列按日期分组后每一组数据的均值，即每列数据每天都有一个均值
    mean = dataframe[dataframe["Type"] == "h"].sort_values("Date", ascending=False).groupby("Date").mean()
    # mean.plot是pandas.tools.plotting.FramePlotMethods,yerr是误差线的值,ylim是y轴上的最小值和最大值。
    # 这里就是画价格均值的曲线图，以价格的方差值作为误差线的值
    mean["Price"].plot(yerr=var["Price"], ylim=(400000, 1500000))
    plt.title("Price mean with Price var yerr")
    plt.show()
    # 计算Type为h且Distance小于13的的按日期分组后数据的均值和标准差，这里是去掉异常值的。
    means = dataframe[(dataframe["Type"] == "h") & (dataframe["Distance"] < 13)].dropna().sort_values("Date",
                                                                                                      ascending=False).groupby(
        "Date").mean()
    errors = dataframe[(dataframe["Type"] == "h") & (dataframe["Distance"] < 13)].dropna().sort_values("Date",
                                                                                                       ascending=False).groupby(
        "Date").std()
    plt.title("means with errors distances over 13")
    means.plot(yerr=errors)
    # 这个图没啥用，因为Price的值太大了，其他的看不出趋势来，如果要看某一列的趋势图，可以按如下方式选择一列来进行绘制
    # means['Rooms'].plot(yerr=errors['Rooms'])
    plt.show()
    dataframe[dataframe["Type"] == "h"].sort_values("Date", ascending=False).groupby("Date").mean()
    pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
    dataframe[(dataframe["Type"] == "h") &
              (dataframe["Distance"] < 14) &
              (dataframe["Distance"] > 13.7)
        # &(dataframe["Suburb"] =="Northcote")
              ].sort_values("Date", ascending=False).dropna().groupby(["Suburb", "SellerG"]).mean()
    sns.kdeplot(dataframe[(dataframe["Suburb"] == "Northcote")
                          & (dataframe["Type"] == "u")
                          & (dataframe["Rooms"] == 2)]["Price"].dropna())
    plt.title("kdeplot Rooms 2")
    plt.show()
    sns.kdeplot(dataframe["Price"][((dataframe["Type"] == "u") &
                                    (dataframe["Distance"] > 3) &
                                    (dataframe["Distance"] < 10) &
                                    (dataframe["Rooms"] > 2)  # &
                                    # (dataframe["Price"] < 1000000)
                                    )].dropna())
    plt.title("kdeplot Rooms More than 2")
    plt.show()
    sns.lmplot("Distance", "Price", dataframe[(dataframe["Rooms"] <= 4) &
                                              (dataframe["Rooms"] > 2) &
                                              (dataframe["Type"] == "h") &
                                              (dataframe["Price"] < 1000000)
                                              ].dropna(), hue="Rooms", size=10)
    plt.title("lmplot Distance Price")
    plt.show()
    dataframe[dataframe["Rooms"] < 4].dropna().groupby(["Distance", "Rooms"]).mean()
    # dataframe.columns
    # pairplot不知道画的什么
    sns.pairplot(dataframe.drop(["Postcode", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
         "CouncilArea", "Address", "Date", "SellerG", "Suburb", "Type", "Method"], axis=1).dropna(), size=5)
    plt.title("pairplot")
    plt.show()
    sns.heatmap(dataframe[dataframe["Type"] == "h"].corr(), annot=True)
    plt.title("heatmap")
    plt.show()
    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    dataframe_dr = dataframe.dropna().sort_values("Date")
    dataframe_dr = dataframe_dr[dataframe_dr["Type"] == "u"]
    all_Data = []
    ##Find out days since start
    days_since_start = [(x - dataframe_dr["Date"].min()).days for x in dataframe_dr["Date"]]
    dataframe_dr["Days"] = days_since_start
    # suburb_dummies = pd.get_dummies(dataframe_dr[["Suburb", "Type", "Method"]])
    suburb_dummies = pd.get_dummies(dataframe_dr[["Type", "Method"]])
    # suburb_dummies = pd.get_dummies(dataframe_dr[[ "Type"]])
    # suburb_dummies = pd.get_dummies(dataframe_dr[["Suburb", "Method"]])
    # all_Data = dataframe_dr.drop(["Address", "Price", "Date", "SellerG", "Suburb", "Type", "Method", "CouncilArea"], axis=1).join(
    #     suburb_dummies)
    all_Data = dataframe_dr.drop(
        ["Postcode", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
         "CouncilArea", "Address", "Price", "Date", "SellerG", "Suburb", "Type", "Method"], axis=1).join(suburb_dummies)
    all_Data.head().to_csv("tony.csv")
    X = all_Data
    y = dataframe_dr["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    ranked_suburbs = coeff_df.sort_values("Coefficient", ascending=False)
    predictions = lm.predict(X_test)
    plt.scatter(y_test, predictions)
    plt.ylim([200000, 1000000])
    plt.xlim([200000, 1000000])
    sns.distplot((y_test - predictions), bins=50)
    plt.title("distplot")
    plt.show()
    from sklearn import metrics
    print("MAE:", metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    plt.show()


if __name__ == '__main__':
    main()
