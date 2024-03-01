### 数据集说明
提供 `20ns` 和 `agnews` 两个新闻文本数据集，每个数据集包含三个文件：

 1. `dataset.csv`: 每行是一篇经过了数据清洗的文本，以及该文本的标签（**注意：在本次弱监督文本分类任务中，文本标签仅用于测试，不能用于训练**）。数据格式如下，
	> [标签ID][,][文本]
	> 例：1,nfl wrap manning wins mvp battle as colts overcome titans peyton manning threw for 254 yards and two touchdowns to win his showdown with fellow co mvp steve mcnair as the indianapolis colts beat the tennessee titans 31 17 in national football league play at nashville on sunday .

 2. `classes.txt`: 每行是一个类别的名称及其ID。数据格式如下，
	> [标签ID][:][标签名称]
	> 例：1:sports

 3. `keywords.txt`: 每行是一个类别的ID及其关键词。数据格式如下，
	 > [标签ID][:][关键词1,关键词2,...]
	 > 例：1:basketball,football,athletes


