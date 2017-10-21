import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import metrics

data = pd.read_csv("/home/jren/Project/CloudPridict/dataNew.csv")

x = data[['MachineNum','Mem/Exe','CPU/Exe','ExeNum/Machine','TotalExeNum','RealCPU/Machine','RealTotalCPU','RealMem/Machine','RealTotalMem']]
y = data[['TIME']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)


def try_different_method(clf,metheName):
    clf.fit(x_train,y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title(metheName+'--'+'Score: %f'%score)
    plt.legend()
    plt.show()



#linear regression
linear_reg = linear_model.LinearRegression()
try_different_method(linear_reg,'LinearRegression')

#Tree regression
from sklearn import tree
tree_reg = tree.DecisionTreeRegressor()
try_different_method(tree_reg,'DecisionTreeRegressor')

from sklearn import svm

svr = svm.SVR()
try_different_method(svr,'SVM')


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor()
try_different_method(knn,'KNN')


from sklearn import ensemble
rf =ensemble.RandomForestRegressor(n_estimators=20) 
try_different_method(rf,'RandomForestRegressor')


ada = ensemble.AdaBoostRegressor(n_estimators=50)
try_different_method(ada,'AdaBoostRegressor')

gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)
try_different_method(gbrt,'GradientBoostingRegressor')
