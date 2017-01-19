import numpy as np

#PATH for training data
train = ""
train_label = ""

#PATH for testing data
test = ""
test_label = ""

file1 = open(train_label, "r")
file2 = open(train, "r")

file3 = open(test_label, "r")
file4 = open(test, "r")

label = list()    # training label set
data = list()     # training data set
check = list()    # if the row will be kept or not

label_test = list()  # similar for testing data
data_test = list()
check_test = list()

# get the number of "dots" in the line
def getDot(list6):
    n = len(list6)
    i = 0
    c = 0
    while(i < n):
        while(i < n and list6[i] == 0):
            i += 1
        if(i == n):
            break
        while(i < n and list6[i] != 0):
            i += 1
        c += 1
    return c

# feature reduction
def filter(list5):
    left = 10 * 28
    right = 17 * 28
    res = list()
    one = [2,2]
    zero = [-1, -2]
    for cur in list5 :
        help = cur[left : right]
        res.append(help)
        '''
        c = getDot(help)
        if(c == 2):
            res.append(zero)
        else:
            res.append(one)
        '''
    return res

i = 0

# filter the data set for training
for line in file1.readlines():
    if (line == "1\n"):
        label.append(1)
        check.append("t")
    elif (line == "0\n"):
        label.append(-1)
        check.append("t")
    else:
        check.append("f")


for line in file2.readlines():
    if (check[i] == 't'):
        line = line.strip('\n')
        row = line.split(" ")
        num = list()
        for ele in row:
            ele = float(ele)
            num.append(ele)
        data.append(num)
    i += 1
data = filter(data)



# filter the testing data set
num_zero = 0
num_one = 0
                                              
for line in file3.readlines():
    if (line == "1\n"):
        num_one += 1
        label_test.append(1)
        check_test.append("t")
    elif (line == "0\n"):
        num_zero += 1
        label_test.append(-1)
        check_test.append("t")
    else:
        check_test.append("f")

k = 0

for line in file4.readlines():
    if (check_test[k] == 't'):
        line = line.strip('\n')
        row = line.split(" ")
        num = list()
        for ele in row:
            ele = float(ele)
            num.append(ele)
        data_test.append(num)
    k += 1

data_test = filter(data_test)


# perceptron algorithm
num = len(data[0])
count = len(data)
initial = 0
w = [initial] * num


def prod(sign, list1):
    res = list()
    for i in list1:
        temp = i * sign
        res.append(temp)
    
    return res

# add the vector
def vectoradd (list1, list2):
    n = len(list1)
    res = list()
    for i in xrange(0, n):
        temp = list1[i] + list2[i]
        res.append(temp)
    
    return res

round = 0
while(round < 10000):
    print round
    round += 1
    j = 0
    total = 0
    while(j < count):
        res = np.dot(w, data[j])
        sign = res * label[j]
        product = prod(label[j], data[j])
        if(sign <= 0):
            w = vectoradd(w, product)

        else :
            total += 1
        j += 1
    
    if(total == count):
        break

    
print w
print round

# calculat the accuracy (number of the mistakes)
def predict(w, data_test, label_test):
    res = list()
    exp = list()
    n = len(data_test)
    zero = 0
    one = 0
    zero_expect = 0
    one_expect = 0
    for i in xrange(n):
        product = np.dot(w, data_test[i])

        if(product == 0 or (product > 0 and label_test[i] < 0) or (product < 0 and label_test[i] > 0)):
            one += 1

    res.append(one)
    return res

result = predict(w, data_test, label_test)

print "number of mistakes is"
print result


