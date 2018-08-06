import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.externals import joblib

path = 'training.data'  # 数据文件路径
data=np.loadtxt(path,dtype=float) #读取数据

x_source, y_source = np.split(data, (6,), axis=1)

scaler = preprocessing.StandardScaler().fit(x_source)  #保存训练集的标准差和均值

path='testing.data'
x_predict=np.loadtxt(path,dtype=float)
y_predict=np.zeros(shape=(len(x_predict),1))
x_predict=scaler.transform(x_predict)
clf = joblib.load('svm_train.pkl')
for i,x in enumerate(x_predict):
	y_predict[i]=clf.predict([x])
result=np.hstack([x_predict,y_predict])
result=pd.DataFrame(result)
result.to_csv('result.csv')
