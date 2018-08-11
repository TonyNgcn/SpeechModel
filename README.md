# SpeechModel

我们总共训练了两个模型，SVM 和 NN。
SVM模型基于scikit-learn 编写，但是在测试集上的准确率大概在70%左右。
SVM模型有两个超参数，经过多次取值，发现γ的取值稳定在7.464263932294464

NN模型基于tensorflow编写，最后一次的训练数据如下：
Saving dict for global step 160000: accuracy = 0.79023886, accuracy_baseline = 0.5109034, auc = 0.8691241, auc_precision_recall = 0.87003034, average_loss = 0.45028788, global_step = 160000, label/mean = 0.5109034, loss = 22.237293, precision = 0.7929293, prediction/mean = 0.49435177, recall = 0.79776424

在测试集上的准确率是79%

综上，我们提交的作业中，result.csv 是由第二种模型得出的
我们把作业一的资料传到了github上，地址为：
Github： https://github.com/TonyNgcn/SpeechModel


