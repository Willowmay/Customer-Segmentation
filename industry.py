# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib
from sklearn import tree
import graphviz 

data = pd.read_csv('hub.csv')
X = data.iloc[:,[2,3,5,6,8,9,10]]
for i in range(X.shape[1]):
    X[X.columns[i]] = pd.factorize(X[X.columns[i]])[0]
Y = data.iloc[:,7]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X.columns,
                                leaves_parallel=True,
                                filled=True) 
graph = graphviz.Source(dot_data) 
graph


#X.to_csv('factorized_x.csv')