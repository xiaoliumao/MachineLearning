import pandas as pd 
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

#读取数据
df = pd.read_csv('data.csv')
#清洗数据
# 把颜色独热编码
df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')
# 把类型独热编码
df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')
# 添加独热编码数据列
df = pd.concat([df, df_colors, df_type], axis=1)
# 去除独热编码对应的原始列
df = df.drop(['Brand', 'Type', 'Color'], axis=1)
# 数据转换
matrix = df.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, square=True)
plt.title('Car Price Variables')
plt.show()
X = df[['Construction Year', 'Days Until MOT', 'Odometer']]
y = df['Ask Price'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

X_normalizer = StandardScaler() # N(0,1)
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train, y_train.ravel())

#Now we can predict prices:
y_pred = knn.predict(X_test)
y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

# Build a plot
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(500, 1500, 100)
plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price')
plt.ylabel('Ask price')
plt.show()

print(y_pred_inv)