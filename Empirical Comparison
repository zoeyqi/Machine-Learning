% Empirical Comparison between different machine learning models
% cross validation

%matplotlib inline
import urllib
import operator
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
In [11]:
'''
Read the dataset and do pre - processing

'''

# url with dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'

# download the file
raw_data = urllib.urlopen(url)

# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")

alldata = dataset[:, :]
for i in alldata :
    i[3] = i[3] - 1

feature_set = dataset[:, 0 : 3]
label_set = dataset[:, 3]
In [12]:
'''
The ROC Curves and AUC for each feature

'''

def feature_ROC(feature_index, alldata) :
    # plot the ROC for each feature
    length = len(alldata)
    auc_list = []
    
    # split the data into 10 folds
    step = length / 10
  
    print("ROC Curve for feature index is:", feature_index)

    plt.plot([0,1],[0,1],"r--",alpha=.5)

    for i in range(0, 10) :
        fpr,tpr,thresh = roc_curve(alldata[(i) * step: (i+1) * step,3], alldata[(i) * step: (i+1) * step,feature_index])
        plt.plot(fpr,tpr)
        auc = roc_auc_score(alldata[(i) * step: (i+1) * step,3], alldata[(i) * step: (i+1) * step,feature_index])
        print(i, "AUC : {}".format(auc))
        auc_list.append(auc)

    plt.show()
    
    print("The mean of AUC is ", np.mean(auc_list))
    print("The standard deviation of AUC is ", np.std(auc_list))
    
In [13]:
'''

The ROC Curves for each of the five algorithms

'''

def algorithm_ROC () :
    # DECISION TREE 
    dt = DecisionTreeClassifier()

    # RAMDOM FOREST
    rf = RandomForestClassifier()

    # GRADIENT BOOSTED TREES
    gb = GradientBoostingClassifier()

    # LOGISTIC REGRESSION
    lr = LogisticRegression()

    # SUPPORT VECTOR MACHINE
    sv = SVC(probability=True)

    tuple = (dt, rf, gb, lr, sv)

    name = ["DT", "RF", "GB", "LR", "SVM"]
    fold = 0
    algo_index = 0

    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    kf.get_n_splits(feature_set)


    for train_index, test_index in kf.split(feature_set):
        X_train, X_test = feature_set[train_index], feature_set[test_index]
        Y_train, Y_test = label_set[train_index], label_set[test_index]
        algo_index = 0
        # train the model for this fold
        for algo in tuple:
            plt.plot([0,1],[0,1],"r--",alpha=.5)
            algo.fit(X_train, Y_train)

            # predict the labels and report accuracy
            hard_pred = algo.predict(X_test)

            acc = np.isclose(hard_pred,Y_test).sum()/len(hard_pred)

            print(fold, name[algo_index], "Accuracy: {}".format(acc))

            # use predicted probabilities to construct ROC curve and AUC score
            soft_pred = algo.predict_proba(X_test)
            fpr,tpr,thresh = roc_curve(Y_test,soft_pred[:,1])

            auc = roc_auc_score(Y_test,soft_pred[:,1])
            plt.plot(fpr,tpr, label = name[algo_index])
            print(fold, name[algo_index], "AUC: {}".format(auc))
            algo_index += 1

        plt.legend(loc = 4)
        plt.show()
        fold += 1
In [14]:
'''

The nested CV for paramter tuning

'''

def nestedCV (algo, parameter) :

    kf_out = KFold(n_splits = 10, random_state = 1, shuffle = False)
    kf_out.get_n_splits(feature_set)
    
    auc_list = []
    
    plt.plot([0,1],[0,1],"r--",alpha=.5)
    for train_out, test_out in kf_out.split(feature_set):
        # get the split for the out loop
        X_train_out, X_test_out = feature_set[train_out], feature_set[test_out]
        Y_train_out, Y_test_out = label_set[train_out], label_set[test_out]

        # choose the best parameter for this fold
        clf = GridSearchCV(estimator = algo, param_grid = parameter, cv = 9)
        clf.fit(X_train_out,Y_train_out)
        print(clf.best_params_)

        # predict the labels and report accuracy
        hard_pred = clf.predict(X_test_out)
        acc = np.isclose(hard_pred,Y_test_out).sum()/len(hard_pred)
        #print("Accuracy: {}".format(acc))

        # use predicted probabilities to construct ROC curve and AUC score
        soft_pred = clf.predict_proba(X_test_out)
        fpr,tpr,thresh = roc_curve(Y_test_out,soft_pred[:,1])
        auc = roc_auc_score(Y_test_out,soft_pred[:,1])
        
        plt.plot(fpr,tpr)
        
        #print("AUC: {}".format(auc))
        auc_list.append(auc)

    plt.show()
    print("The mean of AUC is ", np.mean(auc_list))
    print("The standard deviation of AUC is ", np.std(auc_list))
In [15]:
'''
The ROC Curves and Accuracy, AUC for each feature

'''

for index in range(0, 3) :
    feature_ROC(index, alldata)
('ROC Curve for feature index is:', 0)
(0, 'AUC : 0.537037037037')
(1, 'AUC : 0.666666666667')
(2, 'AUC : 0.583732057416')
(3, 'AUC : 0.471291866029')
(4, 'AUC : 0.738636363636')
(5, 'AUC : 0.565')
(6, 'AUC : 0.496894409938')
(7, 'AUC : 0.78125')
(8, 'AUC : 0.735795454545')
(9, 'AUC : 0.571428571429')

('The mean of AUC is ', 0.61477324266969569)
('The standard deviation of AUC is ', 0.10307318778237698)
('ROC Curve for feature index is:', 1)
(0, 'AUC : 0.691358024691')
(1, 'AUC : 0.680555555556')
(2, 'AUC : 0.397129186603')
(3, 'AUC : 0.555023923445')
(4, 'AUC : 0.488636363636')
(5, 'AUC : 0.535')
(6, 'AUC : 0.527950310559')
(7, 'AUC : 0.480113636364')
(8, 'AUC : 0.338068181818')
(9, 'AUC : 0.329192546584')

('The mean of AUC is ', 0.5023027729255799)
('The standard deviation of AUC is ', 0.11861835037512464)
('ROC Curve for feature index is:', 2)
(0, 'AUC : 0.66049382716')
(1, 'AUC : 0.423611111111')
(2, 'AUC : 0.650717703349')
(3, 'AUC : 0.724880382775')
(4, 'AUC : 0.747159090909')
(5, 'AUC : 0.8225')
(6, 'AUC : 0.847826086957')
(7, 'AUC : 0.636363636364')
(8, 'AUC : 0.741477272727')
(9, 'AUC : 0.751552795031')

('The mean of AUC is ', 0.70065819063835844)
('The standard deviation of AUC is ', 0.11336023337141479)
In [16]:
'''
Get the ROC Curves for 5 algorithms

DT - DECISION TREE
RF - RAMDOM FOREST
GB - GRADIENT BOOSTED TREES
LG - LOGISTIC REGRESSION
SVM - SUPPORT VECTOR MACHINE

'''
algorithm_ROC()
(0, 'DT', 'Accuracy: 0')
(0, 'DT', 'AUC: 0.72619047619')
(0, 'RF', 'Accuracy: 0')
(0, 'RF', 'AUC: 0.782738095238')
(0, 'GB', 'Accuracy: 0')
(0, 'GB', 'AUC: 0.776785714286')
(0, 'LR', 'Accuracy: 0')
(0, 'LR', 'AUC: 0.857142857143')
(0, 'SVM', 'Accuracy: 0')
(0, 'SVM', 'AUC: 0.767857142857')

(1, 'DT', 'Accuracy: 0')
(1, 'DT', 'AUC: 0.669642857143')
(1, 'RF', 'Accuracy: 0')
(1, 'RF', 'AUC: 0.732142857143')
(1, 'GB', 'Accuracy: 0')
(1, 'GB', 'AUC: 0.690476190476')
(1, 'LR', 'Accuracy: 0')
(1, 'LR', 'AUC: 0.654761904762')
(1, 'SVM', 'Accuracy: 0')
(1, 'SVM', 'AUC: 0.607142857143')

(2, 'DT', 'Accuracy: 0')
(2, 'DT', 'AUC: 0.588383838384')
(2, 'RF', 'Accuracy: 0')
(2, 'RF', 'AUC: 0.719696969697')
(2, 'GB', 'Accuracy: 0')
(2, 'GB', 'AUC: 0.641414141414')
(2, 'LR', 'Accuracy: 0')
(2, 'LR', 'AUC: 0.742424242424')
(2, 'SVM', 'Accuracy: 0')
(2, 'SVM', 'AUC: 0.666666666667')

(3, 'DT', 'Accuracy: 0')
(3, 'DT', 'AUC: 0.359523809524')
(3, 'RF', 'Accuracy: 0')
(3, 'RF', 'AUC: 0.538095238095')
(3, 'GB', 'Accuracy: 0')
(3, 'GB', 'AUC: 0.552380952381')
(3, 'LR', 'Accuracy: 0')
(3, 'LR', 'AUC: 0.595238095238')
(3, 'SVM', 'Accuracy: 0')
(3, 'SVM', 'AUC: 0.447619047619')

(4, 'DT', 'Accuracy: 0')
(4, 'DT', 'AUC: 0.483695652174')
(4, 'RF', 'Accuracy: 0')
(4, 'RF', 'AUC: 0.5625')
(4, 'GB', 'Accuracy: 0')
(4, 'GB', 'AUC: 0.559782608696')
(4, 'LR', 'Accuracy: 0')
(4, 'LR', 'AUC: 0.592391304348')
(4, 'SVM', 'Accuracy: 0')
(4, 'SVM', 'AUC: 0.58152173913')

(5, 'DT', 'Accuracy: 0')
(5, 'DT', 'AUC: 0.604761904762')
(5, 'RF', 'Accuracy: 0')
(5, 'RF', 'AUC: 0.714285714286')
(5, 'GB', 'Accuracy: 0')
(5, 'GB', 'AUC: 0.695238095238')
(5, 'LR', 'Accuracy: 0')
(5, 'LR', 'AUC: 0.6')
(5, 'SVM', 'Accuracy: 0')
(5, 'SVM', 'AUC: 0.561904761905')

(6, 'DT', 'Accuracy: 0')
(6, 'DT', 'AUC: 0.52380952381')
(6, 'RF', 'Accuracy: 0')
(6, 'RF', 'AUC: 0.39417989418')
(6, 'GB', 'Accuracy: 0')
(6, 'GB', 'AUC: 0.52380952381')
(6, 'LR', 'Accuracy: 0')
(6, 'LR', 'AUC: 0.608465608466')
(6, 'SVM', 'Accuracy: 0')
(6, 'SVM', 'AUC: 0.518518518519')

(7, 'DT', 'Accuracy: 0')
(7, 'DT', 'AUC: 0.65')
(7, 'RF', 'Accuracy: 0')
(7, 'RF', 'AUC: 0.7475')
(7, 'GB', 'Accuracy: 0')
(7, 'GB', 'AUC: 0.67')
(7, 'LR', 'Accuracy: 0')
(7, 'LR', 'AUC: 0.78')
(7, 'SVM', 'Accuracy: 0')
(7, 'SVM', 'AUC: 0.755')

(8, 'DT', 'Accuracy: 0')
(8, 'DT', 'AUC: 0.6')
(8, 'RF', 'Accuracy: 0')
(8, 'RF', 'AUC: 0.848')
(8, 'GB', 'Accuracy: 0')
(8, 'GB', 'AUC: 0.824')
(8, 'LR', 'Accuracy: 0')
(8, 'LR', 'AUC: 0.792')
(8, 'SVM', 'Accuracy: 0')
(8, 'SVM', 'AUC: 0.84')

(9, 'DT', 'Accuracy: 0')
(9, 'DT', 'AUC: 0.541666666667')
(9, 'RF', 'Accuracy: 0')
(9, 'RF', 'AUC: 0.576388888889')
(9, 'GB', 'Accuracy: 0')
(9, 'GB', 'AUC: 0.5')
(9, 'LR', 'Accuracy: 0')
(9, 'LR', 'AUC: 0.708333333333')
(9, 'SVM', 'Accuracy: 0')
(9, 'SVM', 'AUC: 0.576388888889')

In [17]:
'''

The nested CV for parameter tuning part

The three algorithms chosen :

1.GRADIENT BOOSTED TREES

2.LOGISTIC REGRESSION

3.SUPPORT VECTOR MACHINE

'''

# 1.GRADIENT BOOSTED TREES
gb = GradientBoostingClassifier()
para_gb = {'n_estimators':range(5, 10, 20)}

nestedCV(gb, para_gb)

# 2.LOGISTIC REGRESSION
lr = LogisticRegression()
para_lr = [{'C': [100, 500, 750, 1000, 3000]}]

nestedCV(lr, para_lr)

# 3.SUPPORT VECTOR MACHINE
sv = SVC(probability=True)
para_sv = [{'C': [1, 10, 100, 1000, 10000]}]

nestedCV(sv, para_sv)
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}
{'n_estimators': 5}

('The mean of AUC is ', 0.6088275685548703)
('The standard deviation of AUC is ', 0.20913934526266292)
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 100}
{'C': 500}

('The mean of AUC is ', 0.69046548276061004)
('The standard deviation of AUC is ', 0.14680564790522982)
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}
{'C': 1}

('The mean of AUC is ', 0.62431828830970715)
('The standard deviation of AUC is ', 0.11563892063020043)
