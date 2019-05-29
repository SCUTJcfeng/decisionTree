# python 3.6

import pydotplus
from common.utils import read_csv
from sklearn import tree, preprocessing
from sklearn.feature_extraction import DictVectorizer


filename = './film.csv'
headers = ['id', 'type', 'country', 'gross', 'watch']
result_list = [data['watch'] for data in read_csv(filename, headers, filter_item_list=['id', 'type', 'country', 'gross'])]
feature_list = [data for data in read_csv(filename, headers, filter_item_list=['id', 'watch'])]


# 转换为array类型的变量, X为特征变量(训练集), Y为结果集
vec = DictVectorizer()
# array([[0., 0., 1., 0., 0., 1., 0., 1., 0.], ....]) 4 2 3 country gross type
dummyX = vec.fit_transform(feature_list).toarray(order=True)

dummyY = preprocessing.LabelBinarizer().fit_transform(result_list)


# criterion 信息熵类型
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
fit_result = clf.fit(dummyX, dummyY)


dot_data = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), filled=True, rounded=True, special_characters=True, out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)

# 画图 依赖graphviz, 需到官网下载msi安装, 设置环境变量
graph.write_pdf('film.pdf')

# 按字母排序 前4位(America China, France, japan) 中间2位(high, low) 最后3位(action, anime, science)
A = ([[0, 0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 1]])  # japan low anime

print(f'predict: {clf.predict(A)}')
