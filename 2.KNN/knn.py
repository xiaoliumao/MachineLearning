#调用KNN函数来实现分类
#读取相应的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.metrics import accuracy_score
#读取数据
iris = datasets.load_iris()
X = iris.data 
y = iris.target 
#print(X,y)
#把数据分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)
# 构建KNN模型， K值为3、 并做训练
clf = KNeighborsClassifier(n_neighbors=3)
#必须要有这一步
clf.fit(X_train, y_train)
# 计算准确率
correct = np.count_nonzero((clf.predict(X_test)==y_test)==True)
print ("Accuracy is: %.3f" %(correct/len(X_test)))