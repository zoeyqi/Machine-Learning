 % Implementation of a “hard” maximum-margin SVM classifier and kernel
 %matplotlib inline
import urllib
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
In [273]:
# url with dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# download the file
raw_data = urllib.urlopen(url)

# pre-processing the data - get the first 100 lines
data = []
k = 0
for i in raw_data:
    t = i.split(',')
    if(k < 50):
        t[len(t) - 1] = '1'
    else:
        t[len(t) - 1] = '-1'
    data.append(map(float, t))
    k = k + 1   
data = data[0:100]

# convert list to a matrix
data_set = np.zeros(shape = (100, 5))
for i in range(0, 100) :
    data_set[i] = data[i]

# divide into feature and lable
feature = data_set[:, 0 : 4]
label = data_set[:, 4]
In [274]:
def matrP(feature, label) :
    P = np.zeros(shape = (100, 100))
    for i in range(0, 100) :
        for j in range(0, 100) :
            P[i][j] = label[i] * label[j] * np.dot(feature[i], feature[j])
    return matrix(P, tc = 'd')
In [275]:
def train() :
    # get the input paramters for the QP solover
    P = matrP(feature, label)

    q = np.zeros(shape = (100, 1))
    for i in range(0, 100) :
        q[i][0] = -1
    q = matrix(q, tc = 'd')

    a = np.zeros(shape = (102, 100))
    for i in range(0, 100) :
        a[0][i] = -label[i]
        a[1][i] = label[i]
        a[2 + i][i] = -1  

    G = matrix(a, tc = 'd')

    h = np.zeros(shape = (102,1))
    h = matrix(h, tc = 'd')

    # get the alpha solutions
    sol = solvers.qp(P, q, G, h)

    # get the parameters for f(x)
    lamda = [0, 0, 0, 0]

    for i in range(0, 100) :
        for j in range(0, 4) :
            lamda[j] = lamda[j] + feature[i][j] * label[i] * sol['x'][i]

    lamda_zero = 1 - np.dot(lamda, feature[0])

    for i in range(1, 100) :
        if(label[i] == 1) :
            lamda_zero = max(lamda_zero, 1 - np.dot(lamda, feature[i]))
    
    return lamda, float(lamda_zero)
In [276]:
def predict(lamda, lamda_zero, x) :
    res = np.dot(lamda, x) + lamda_zero
    if(res > 0) :
        return 1
    return -1
In [277]:
# get the prediction function from train()
lamda, lamda_zero = train()

# get the prediction results using predict function
class1 = predict(lamda, lamda_zero, feature[14])
class2 = predict(lamda, lamda_zero, feature[88])

# expect to be 1
print res1

# expect to be -1
print res2
     pcost       dcost       gap    pres   dres
 0: -3.9652e+00 -6.9071e+00  2e+02  2e+01  2e+00
 1: -1.3873e+00 -2.0763e+00  2e+01  1e+00  1e-01
 2: -4.1276e-01 -1.4485e+00  3e+00  1e-01  2e-02
 3: -4.4907e-01 -8.5958e-01  5e-01  4e-03  4e-04
 4: -5.7377e-01 -7.6490e-01  2e-01  1e-03  2e-04
 5: -6.5782e-01 -7.5988e-01  1e-01  4e-04  4e-05
 6: -7.4122e-01 -7.4929e-01  8e-03  2e-05  2e-06
 7: -7.4792e-01 -7.4808e-01  2e-04  3e-07  3e-08
 8: -7.4805e-01 -7.4806e-01  3e-06  4e-09  3e-09
 9: -7.4806e-01 -7.4806e-01  4e-08  4e-11  4e-09
Optimal solution found.
1
-1
