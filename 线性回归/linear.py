'''
机器学习之线性回归
已有数据是：电视广告的投入(x)、产品销售量(y)
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
import os 
import pickle
#保存模型
from sklearn.externals import joblib
#读取csv文件
data = pd.read_csv('data/Advertising.csv')
#显示出前5条数据
print(data.head())
#每列是什么
print(data.columns)
#通过数据可视化分析数据
# plt.figure(figsize=(16,9))
# plt.scatter(data['TV'],data['sales'],c='red')
# plt.xlabel('Money spent on TV ads')
# plt.ylabel('Sales')
# plt.show()
#训练线性回归模型
X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
'''
coef_ 存放回归系数
intercept_则存放截距
'''
reg = LinearRegression()
reg.fit(X,y)
#保存模型
s = pickle.dumps(reg)
#恢复模型
clf2 = pickle.loads(s)
#预测
print(clf2.predict([[200]]))
print('投入一亿元的电视广告, 预计的销售量为{:.5}亿'.format(clf2.predict([[200]])[0][0]))

print(reg.coef_[0][0])
print('a = {:.5}'.format(reg.coef_[0][0]))
print('b = {:.5}'.format(reg.intercept_[0]))
print('线性模型为：Y = {:.5}X + {:.5}'.format(reg.coef_[0][0], reg.intercept_[0]))
#可视化训练好的线性回归方程模型
'''
predict(X)：预测方法，将返回预测值y_pred
'''
predictions = reg.predict(X)
plt.figure(figsize=(16,9))
plt.scatter(data['TV'], data['sales'], c ='black')
#画那一条线
plt.plot(data['TV'], predictions,c ='blue', linewidth=2)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.show()
#做预测
# predictions = reg.predict([[100]])
# print('投入一亿元的电视广告, 预计的销售量为{:.5}亿'.format( predictions[0][0]) )
