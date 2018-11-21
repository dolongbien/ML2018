import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydot
from io import StringIO
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#df = pd.read_csv("tennis.csv", delimiter="\t")
df = pd.read_csv("bar.csv", delimiter=",")

attributeNames = [v for v in df.head(0)]

className = attributeNames.pop(-1)
features = attributeNames


dtree = DecisionTreeClassifier(criterion = "entropy")

X = pd.get_dummies(df[attributeNames],drop_first=True)
Y = df[className]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
dtree.fit(X, Y)
dotfile = StringIO()
tree.export_graphviz(dtree, out_file=dotfile, class_names=dtree.classes_)
(graph,) = pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("dtree.png")