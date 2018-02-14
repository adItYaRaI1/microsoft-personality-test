# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:44:39 2018

@author: Aditya Rai
"""

import re
import numpy as np
import collections
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('mbti_1.csv')
print(df.head(10))
print(df.info())
df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)

#plt.figure(figsize=(15,10))
#sns.violinplot(x='type', y='words_per_comment', data=df, inner=None, color='lightgray')
#sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
#plt.ylabel('Words per comment')
#plt.show()

df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)

#plt.figure(figsize=(15,10))
#sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df, kind='kde')

#i = df['type'].unique()
#k = 0
#TypeArray = []
#PearArray=[]
#for m in range(0,2):
#    for n in range(0,6):
#        df_2 = df[df['type'] == i[k]]
#        pearsoncoef1=np.corrcoef(x=df_2['words_per_comment'], y=df_2['ellipsis_per_comment'])
#        pear=pearsoncoef1[1][0]
#        print(pear)
#        TypeArray.append(i[k])
#        PearArray.append(pear)
#        k+=1
#
#TypeArray = [x for _,x in sorted(zip(PearArray,TypeArray))]
#PearArray = sorted(PearArray, reverse=True)
#print(PearArray)
#print(TypeArray)
#plt.scatter(TypeArray, PearArray)


map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)
print(df.head(10))

X = df.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
y = df['type'].values

print(y.shape)
print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.1, random_state=5)

#sgd = SGDClassifier(max_iter=5, tol=None)
#sgd.fit(X_train, y_train)
#Y_pred = sgd.predict(X_test)
#sgd.score(X_train, y_train)
#acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
#print(round(acc_sgd,2,), "%")

X = df.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
y = df['type'].values

print(y.shape)
print(X.shape)

X_train,X_test,y_train, y_test=train_test_split(X,y,test_size = 0.0, random_state=5)

#sgd = SGDClassifier(max_iter=5, tol=None)
#sgd.fit(X_train, y_train)
#Y_pred = sgd.predict(X_test)
#sgd.score(X_train, y_train)
#acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
#print(round(acc_sgd,2,), "%")

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

#Y_prediction = random_forest.predict(X_test)
#
#random_forest.score(X_train, y_train)
#acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
#print(round(acc_random_forest,2,), "%")

#xt = df[200:300]
#xxt = xt.drop(['posts','I-E','N-S','T-F','J-P'], axis=1).values
#tt = xt.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
#xt_prediction = random_forest.predict(tt)
#xt['pred'] = xt_prediction
#
#testonerow = df[1:2]
#testonerow1 = testonerow.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
#testonerow_pred = random_forest.predict(testonerow1)
#print(testonerow_pred)

test_data1 = pd.read_csv('test_data - test_data.csv')
test_data1 = test_data1[['content']]

test_data1['words_per_comment'] = test_data1['content'].apply(lambda x: len(x.split())/50)
test_data1['http_per_comment'] = test_data1['content'].apply(lambda x: x.count('http')/50)
test_data1['music_per_comment'] = test_data1['content'].apply(lambda x: x.count('music')/50)
test_data1['question_per_comment'] = test_data1['content'].apply(lambda x: x.count('?')/50)
test_data1['img_per_comment'] = test_data1['content'].apply(lambda x: x.count('jpg')/50)
test_data1['excl_per_comment'] = test_data1['content'].apply(lambda x: x.count('!')/50)
test_data1['ellipsis_per_comment'] = test_data1['content'].apply(lambda x: x.count('...')/50)
test_data2 = test_data1.drop(['content'], axis=1).values
testonerow_pred = random_forest.predict(test_data2)
print(testonerow_pred)
test_data1['predicted'] =  testonerow_pred

finalexport = test_data1[['content','predicted']]
finalexport.to_csv('file_name11.csv' , index=False)
