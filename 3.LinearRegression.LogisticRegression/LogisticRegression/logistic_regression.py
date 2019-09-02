import pandas as pd 
import numpy as np 
from sklearn import preprocessing 
import matplotlib.pyplot as plt 
plt.rc('font',size=14)
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
import seaborn as sns 
sns.set(style='white')
sns.set(style='whitegrid',color_codes=True)

data = pd.read_csv('banking.csv',header=0)
data = data.dropna()
# print(data.shape)
# print(list(data.columns))
(data['education'].unique())
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y','Basic',data['education'])
data['education'] = np.where(data['education']=='basic.4y','Basic',data['education'])

#查看label的数量
#print(data['y'].value_counts())
# sns.countplot(x='y',data=data,palette='hls')
# plt.show()
# plt.savefig('count_plot')
data.groupby('y').mean()
