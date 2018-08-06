import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib

path = 'training.data'  # 数据文件路径
data=np.loadtxt(path,dtype=float) #读取数据
np.random.shuffle(data) #打乱数据

#划分数据,分成训练集，测试集，验证集
x, y = np.split(data, (6,), axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.9)
x_test, x_validation, y_test, y_validation=train_test_split(x_test, y_test, random_state=0, train_size=0.3)

#数据归一化
scaler = preprocessing.StandardScaler().fit(x_train)  #保存训练集的标准差和均值
x_train=scaler.transform(x_train) #训练集数据归一化
x_test=scaler.transform(x_test)   #测试集使用训练集数据归一化
x_validation=scaler.transform(x_validation) #验证集使用训练集数据归一化

#从2^-5-2^15区间中寻找最优C值，从2^-15-2^3区间寻找最优γ值
CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
#gammaScale = [-15, -13, -11, -9, -7, -5,-3,-1,1,3]
score={'c':0.0,'g':0.0,'score':0.0}
for Cindex,c in enumerate(CScale):
#	for Gindex,g in enumerate(gammaScale):
		#clf = svm.SVC(C=pow(2,c), kernel='rbf', gamma=pow(2,g), decision_function_shape='ovo')
		clf = svm.SVC(C=pow(2,c), kernel='rbf', gamma=7.464263932294464, decision_function_shape='ovo')
		clf.fit(x_train, y_train.ravel())
		#result=clf.score(x_train, y_train)
		result=clf.score(x_test, y_test)
		if result>score['score']:
			score['c']=Cindex
			#score['g']=Gindex
			score['score']=result


n = 10
minCScale = 0.5*(CScale[int(max(0,score['c']-1))]+CScale[int(score['c'])])
maxCScale = 0.5*(CScale[int(min(len(CScale)-1,score['c']+1))]+CScale[int(score['c'])])
newCScale=np.arange(minCScale,maxCScale,(maxCScale-minCScale)/n)
print(newCScale)
#mingammaScale = 0.5*(gammaScale[int(max(0,score['g']-1))]+gammaScale[int(score['g'])])
#maxgammaScale = 0.5*(gammaScale[int(min(len(gammaScale)-1,score['g']+1))]+gammaScale[int(score['g'])])
#newgammaScale=np.arange(mingammaScale,maxgammaScale,(maxgammaScale-mingammaScale)/n)
#print(newgammaScale)

score['score']=0.0
for c in newCScale:
	#for g in newgammaScale:
		clf = svm.SVC(C=pow(2, c), kernel='rbf', gamma=7.464263932294464, decision_function_shape='ovo')
		#clf = svm.SVC(C=pow(2,c), kernel='rbf', gamma=pow(2,g), decision_function_shape='ovo')
		clf.fit(x_train, y_train.ravel())
		#result=clf.score(x_train, y_train)
		result=clf.score(x_test, y_test)
		if result>score['score']:
			score['c']=pow(2,c)
			#score['g']=pow(2,g)
			score['score']=result


#print('bestC:'+str(score['c'])+' bestgamma:'+str(score['g']))
#print('bestC:'+str(score['c']))
#bestC:1024.0 bestgamma:7.464263932294464
#bestC:16.0 bestgamma:6.498019170849887
#bestC:32.00000000000002 bestgamma:7.464263932294464
#C=31216.041919903313(local)
#bestC 0.8705505632961239(colab)
#bestC 0.5743491774985174(colab)
#clf = svm.SVC(C=score['c'], kernel='rbf', gamma=7.464263932294464, decision_function_shape='ovo')
#clf = svm.SVC(C=score['c'], kernel='rbf', gamma=score['g'], decision_function_shape='ovo')

clf = svm.SVC(C=31216.041919903313, kernel='rbf', gamma=7.464263932294464, decision_function_shape='ovo')
clf.fit(x_train, y_train.ravel())
#保存模型
joblib.dump(clf, 'svm_train.pkl')
#加载模型
#clf = joblib.load('svm_train.pkl')
#查看结果
print("训练集精度",clf.score(x_train, y_train))
print("测试集精度",clf.score(x_test, y_test))
print("验证集精度",clf.score(x_validation, y_validation))
