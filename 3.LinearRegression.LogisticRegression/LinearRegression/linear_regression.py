import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime as dt
from plotly import tools
from plotly.graph_objs import*
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
import chart_studio.plotly as py
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing



df = pd.read_csv('./000001.csv')
'''
股票数据的特征
data 日期
open 开盘价
close 收盘价
volume 成交量
price_change 价格变动
p_change 涨跌幅
ma5 5日均价
ma20 20日均价
v_ma5 5日均量
v_ma10 10日均量
v_ma20 20日均量
'''
#将每一个数据的键值的类型从字符串转为日期
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
#按时间升序排序
df.sort_values(by=['date'],inplace=True,ascending=True)
#检测是否有缺失数据 NaNs
df.dropna(axis=0,inplace=True)
#print(df.isna().sum())

# Min_date = df.index.min()
# Max_date = df.index.max()
# print('first date is',Min_date)
# print("Last date is",Max_date)
# print(Max_date - Min_date)
# trace = go.Ohlc(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])
# data = [trace]
# iplot(data, filename='simple_ohlc')

#线性回归
# 创建新的列、包含预测值。根据当前的数据预测5天以后的收盘价
num = 5 # 预测5天后的情况
df['label'] = df['close'].shift(-num)

#丢弃'label','price_change','p_change',不需要他们做预测
Data = df.drop(['label','price_change','p_change'],axis = 1)
# print(Data.tail())
X = Data.values
X = preprocessing.scale(X)
X = X[:-num]
df.dropna(inplace=True)
Target = df.label
y = Target.values

# 将数据分为训练数据和测试数据
X_train, y_train = X[0:550, :], y[0:550]
X_test, y_test = X[550:, -51:], y[550:606]

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test) # 使用绝对系数 R^2 评估模型

# 做预测 

X_Predict = X[-num:]
Forecast = lr.predict(X_Predict)