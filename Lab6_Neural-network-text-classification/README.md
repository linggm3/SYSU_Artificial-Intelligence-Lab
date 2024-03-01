Code 存放 python 程序文件
	processed_data  存放已经处理好的数据
	data_preprocess.py  对原始数据进行文本清洗，结果放到 processed_data 中
	cnn_train.py  CNN 训练函数
	cnn_test.py  CNN 测试函数（加载 Result 中的模型，读取测试集，进行预测）
	rnn_train.py  RNN 训练函数
	rnn_test.py  RNN 测试函数（加载 Result 中的模型，读取测试集，进行预测）

Result 存放运行的结果
	21307077_lingguoming_CNN_classification.csv 存放 CNN 在测试集上的表现
	21307077_lingguoming_RNN_classification.csv 存放 RNN 在测试集上的表现
	两者的格式都是 label '\t' text

Report
	精心杜撰的报告

NOTICE
	为防止压缩包过大，glove.6B.50d 文件已被我删除
	如果您要跑我的代码，请您将 glove.6B.50d 文件加到 Code 文件夹中