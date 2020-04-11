#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: predicting_H1B_status.py
@time: 2018/8/24 9:32 AM
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from statistics import mode
import re
import xgboost
import pickle
import warnings
warnings.filterwarnings("ignore")

"""
In [6]: df = pd.read_csv("/Users/yuxuecheng/Documents/ml_data/h1b_kaggle.csv")
In [8]: df.columns
Out[8]:
Index(['Unnamed: 0', 'CASE_STATUS', 'EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE',
       'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR', 'WORKSITE', 'lon',
       'lat'],
      dtype='object')

In [9]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3002458 entries, 0 to 3002457
Data columns (total 11 columns):
Unnamed: 0            int64
CASE_STATUS           object
EMPLOYER_NAME         object
SOC_NAME              object
JOB_TITLE             object
FULL_TIME_POSITION    object
PREVAILING_WAGE       float64
YEAR                  float64
WORKSITE              object
lon                   float64
lat                   float64
dtypes: float64(4), int64(1), object(6)
memory usage: 252.0+ MB
In [12]: df.head()
Out[12]:
   Unnamed: 0          CASE_STATUS                                      EMPLOYER_NAME                       SOC_NAME    ...        YEAR                 WORKSITE         lon        lat
0           1  CERTIFIED-WITHDRAWN                             UNIVERSITY OF MICHIGAN  BIOCHEMISTS AND BIOPHYSICISTS    ...      2016.0      ANN ARBOR, MICHIGAN  -83.743038  42.280826
1           2  CERTIFIED-WITHDRAWN                             GOODMAN NETWORKS, INC.               CHIEF EXECUTIVES    ...      2016.0             PLANO, TEXAS  -96.698886  33.019843
2           3  CERTIFIED-WITHDRAWN                          PORTS AMERICA GROUP, INC.               CHIEF EXECUTIVES    ...      2016.0  JERSEY CITY, NEW JERSEY  -74.077642  40.728158
3           4  CERTIFIED-WITHDRAWN  GATES CORPORATION, A WHOLLY-OWNED SUBSIDIARY O...               CHIEF EXECUTIVES    ...      2016.0         DENVER, COLORADO -104.990251  39.739236
4           5            WITHDRAWN                          PEABODY INVESTMENTS CORP.               CHIEF EXECUTIVES    ...      2016.0      ST. LOUIS, MISSOURI  -90.199404  38.627003

[5 rows x 11 columns]

In [13]: df.describe()
Out[13]:
         Unnamed: 0  PREVAILING_WAGE          YEAR           lon           lat
count  3.002458e+06     3.002373e+06  3.002445e+06  2.895216e+06  2.895216e+06
mean   1.501230e+06     1.469984e+05  2.013855e+03 -9.213441e+01  3.816054e+01
std    8.667351e+05     5.287609e+06  1.680612e+00  1.965591e+01  4.672835e+00
min    1.000000e+00     0.000000e+00  2.011000e+03 -1.578583e+02  1.343719e+01
25%    7.506152e+05     5.437100e+04  2.012000e+03 -1.119261e+02  3.416536e+01
50%    1.501230e+06     6.502100e+04  2.014000e+03 -8.615862e+01  3.910312e+01
75%    2.251844e+06     8.143200e+04  2.015000e+03 -7.551381e+01  4.088374e+01
max    3.002458e+06     6.997607e+09  2.016000e+03  1.457298e+02  6.483778e+01
"""

df = pd.read_csv("/Users/yuxuecheng/Documents/ml_data/h1b_kaggle.csv")
df.CASE_STATUS[df['CASE_STATUS'] == 'REJECTED'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS'] == 'INVALIDATED'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS'] ==
               'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS'] == 'CERTIFIED-WITHDRAWN'] = 'CERTIFIED'

# Drop rows with withdrawn
df.EMPLOYER_NAME.describe()
df = df.drop(df[df.CASE_STATUS == 'WITHDRAWN'].index)

# Storing non null in df w.r.t. case status
df = df[df['CASE_STATUS'].notnull()]
print(df['CASE_STATUS'].value_counts())

# Filling na in employer name with mode
df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].fillna(df['EMPLOYER_NAME'].mode()[0])

assert pd.notnull(df['EMPLOYER_NAME']).all().all()

# to check the percentile in wages
print(np.nanpercentile(df.PREVAILING_WAGE, 98))
df.PREVAILING_WAGE.median()

# replacing min and max with 2 and 98 percentile
df.loc[df.PREVAILING_WAGE < 34029, 'PREVAILING_WAGE'] = 34029
df.loc[df['PREVAILING_WAGE'] > 138703, 'PREVAILING_WAGE'] = 138703
df.PREVAILING_WAGE.fillna(df.PREVAILING_WAGE.mean(), inplace=True)

# Filling na in JOB_TITLE and FULL_TIME_POSITION with mode
df['JOB_TITLE'] = df['JOB_TITLE'].fillna(df['JOB_TITLE'].mode()[0])
df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna(
    df['FULL_TIME_POSITION'].mode()[0])
df['SOC_NAME'] = df['SOC_NAME'].fillna(df['SOC_NAME'].mode()[0])

foo1 = df['FULL_TIME_POSITION'] == 'Y'
foo2 = df['CASE_STATUS'] == 'CERIFIED'
print(len(df[foo1])/len(df))

fooy = df.FULL_TIME_POSITION[df['FULL_TIME_POSITION'] == 'Y'].count()
foox = df.CASE_STATUS[df['CASE_STATUS'] == 'CERIFIED'].count()
print(fooy/df.FULL_TIME_POSITION.count())

# Dropping lat and lon columns
df = df.drop('lat', axis=1)
df = df.drop('lon', axis=1)

df['NEW_EMPLOYER'] = np.nan
df.shape

warnings.filterwarnings("ignore")

df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].str.lower()
df.NEW_EMPLOYER[df['EMPLOYER_NAME'].str.contains('university')] = 'university'
df['NEW_EMPLOYER'] = df.NEW_EMPLOYER.replace(
    np.nan, 'non university', regex=True)

# Creating occupation and mapping the values
warnings.filterwarnings("ignore")

df['OCCUPATION'] = np.nan
df['SOC_NAME'] = df['SOC_NAME'].str.lower()
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'computer', 'programmer')] = 'computer occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'software', 'web developer')] = 'computer occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('database')] = 'computer occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'math', 'statistic')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'predictive model', 'stats')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'teacher', 'linguist')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'professor', 'Teach')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'school principal')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'medical', 'doctor')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'physician', 'dentist')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'Health', 'Physical Therapists')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'surgeon', 'nurse')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('psychiatr')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'chemist', 'physicist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'biology', 'scientist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'biologi', 'clinical research')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'public relation', 'manage')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'management', 'operation')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'chief', 'plan')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'executive')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'advertis', 'marketing')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'promotion', 'market research')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'business', 'business analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'business systems analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'accountant', 'finance')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'financial')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'engineer', 'architect')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'surveyor', 'carto')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'technician', 'drafter')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains(
    'information security', 'information tech')] = 'Architecture & Engineering'
df['OCCUPATION'] = df.OCCUPATION.replace(np.nan, 'Others', regex=True)

# Splitting city and state and capturing state in another variable
df['state'] = df.WORKSITE.str.split('\s+').str[-1]

print(df.head())

class_mapping = {'CERTIFIED': 0, 'DENIED': 1}
df["CASE_STATUS"] = df["CASE_STATUS"].map(class_mapping)

print(df.head())

test1 = pd.Series(df['JOB_TITLE'].ravel()).unique()
print(pd.DataFrame(test1))

# dropping these columns
df = df.drop('EMPLOYER_NAME', axis=1)
df = df.drop('SOC_NAME', axis=1)
df = df.drop('JOB_TITLE', axis=1)
df = df.drop('WORKSITE', axis=1)
df = df.drop('CASE_ID', axis=1)

df1 = df.copy()

df1[['CASE_STATUS', 'FULL_TIME_POSITION', 'YEAR', 'NEW_EMPLOYER', 'OCCUPATION', 'state']] = df1[['CASE_STATUS',
                                                                                                 'FULL_TIME_POSITION', 'YEAR', 'NEW_EMPLOYER', 'OCCUPATION', 'state']].apply(lambda x: x.astype('category'))
df1.info()

X = df.drop('CASE_STATUS', axis=1)
y = df.CASE_STATUS

seed = 7
test_size = 0.40
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)
X_train.columns

print(X_train.isnull().sum())

X_train_encode = pd.get_dummies(X_train)
X_test_encode = pd.get_dummies(X_test)
y_train.head()

train_X = X_train_encode.as_matrix()
train_y = y_train.as_matrix()

gbm = xgboost.XGBClassifier(
    max_features='sqrt', subsample=0.8, random_state=10)

parameters = [{'n_estimators': [10, 100]},
              {'learning_rate': [0.1, 0.01, 0.5]}]
grid_search = GridSearchCV(
    estimator=gbm, param_grid=parameters, scoring='accuracy', cv=3, n_jobs=-1)
grid_search = grid_search.fit(train_X, train_y)
warnings.filterwarnings("ignore")

grid_search.grid_scores_,
grid_search.best_params_,
grid_search.best_score_
grid_search.best_estimator_

gbm = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=0,
                            max_depth=3, max_features='sqrt', min_child_weight=1, missing=None,
                            n_estimators=100, n_jobs=1, nthread=None,
                            objective='binary:logistic', random_state=10, reg_alpha=0,
                            reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
                            subsample=0.8).fit(train_X, train_y)

y_pred = gbm.predict(X_test_encode.as_matrix())
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

roc_auc_score(y_test, y_pred)

fpr_xg, tpr_xg, thresholds = metrics.roc_curve(y_test, y_pred)
print(metrics.auc(fpr_xg, tpr_xg))
auc_xgb = np.trapz(tpr_xg, fpr_xg)
plt.plot(fpr_xg, tpr_xg, label=" auc="+str(auc_xgb))
plt.legend(loc=4)
plt.show()

"""Saving the Model"""
XGB_Model_h1b = 'XGB_Model_h1b.sav'
pickle.dump(gbm, open(XGB_Model_h1b, 'wb'))
