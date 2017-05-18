import numpy as np
import platform
import time
from pyspark import SparkConf, SparkContext

#sigmoid function
#when deriv==true, return the derivative function value instead.
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def train(node_num, X, y, syn):
    l = [[] for _ in range(len(node_num))]
    # l0 is the X
    l[0] = X
    # calculate the value on layer1 & y, with non-linear(sigmoid) function
    for i in range(0,len(node_num)-1):
        l[i+1] = nonlin(np.dot(l[i], syn[i]))

    softmax = [0 for _ in range(node_num[-1])]
    for i in range(0, node_num[-1]):
        z = pow(np.e, l[-1][0][i])
        softmax[i] = z
    sum_sft = sum(softmax)
    for i in range(0, node_num[-1]):
        softmax[i] = softmax[i] / sum_sft

    error = delta = [0 for _ in range(len(node_num))]
    for i in xrange(len(node_num)-1,0,-1):
        if i == len(node_num)-1:
            error[i] = y - softmax
        else:
            error[i] = delta[i+1].dot(syn[i].T)
        delta[i] = error[i] * nonlin(l[i], deriv=True)

    ret = []
    for i in range(0,len(node_num)-1):
        ret.append(l[i].T.dot(delta[i+1]))
    ret.append(error[-1])

    return ret

def classify(node_num, X, syn):
    l = [[] for _ in range(len(node_num))]
    l[0] = X
    for i in range(0,len(node_num)-1):
        l[i+1] = nonlin(np.dot(l[i], syn[i]))

    softmax = [0 for _ in range(node_num[-1])]
    for i in range(0, node_num[-1]):
        z = pow(np.e, l[-1][0][i])
        softmax[i] = z
    sum_sft = sum(softmax)
    for i in range(0, node_num[-1]):
        softmax[i] = softmax[i] / sum_sft

    for i in range(0,node_num[-1]):
        if softmax[i] == max(softmax):
            return i

np.random.seed(1)

#init spark
conf = (SparkConf()
        .setAppName("SPARK_ANN")
        .setMaster("spark://192.168.0.3:7077"))
#        .setMaster("local[*]"))
sc = SparkContext(conf=conf)

#load from file
if platform.system() == 'Linux':
    path = '/home/master/Downloads/WISDM_at_v2.0/WISDM_at_v2.0_raw.txt'
elif platform.system() == 'Windows':
    path = 'C:\Users\KUsch\Downloads\WISDM_at_v2.0\WISDM_at_v2.0_raw.txt'
else:
    path = '/Users/Abj/Downloads/WISDM_at_v2.0/WISDM_at_v2.0_raw.txt'

csv = sc.textFile(path,24)
data = csv.filter(lambda line: line[-1]==';')\
    .map(lambda line: (line.split(",")))\
    .filter(lambda line: len(line)==6)
#replace exercise name to num
def replace(line):
    exercise = {"Walking" : 0,"Jogging" : 1,"Stairs" : 2,"Sitting" : 3,"Standing" : 4,"LyingDown" : 5}
    line[1] = exercise[line[1]]
    line[5] = line[5].replace(";","")
    return line
data = data.map(lambda line: replace(line))
#form change to (X,Y,Z), result
def change(line):
    temp = [0.0 for _ in range(6)]
    temp[line[1]] = 1.0
    return ((float(line[3]),float(line[4]),float(line[5])),temp)
train_data = data.map(lambda line: change(line))
#shuffle rdd
train_data = train_data.repartition(24)

#first = 3, last = 6
node_num = [3,1000,6]
num_of_train = 100
batch_size = 100

syn = []
for i in range(0,len(node_num)-1):
    syn.append(np.random.random((node_num[i], node_num[i+1])))

train_data = train_data.zipWithIndex()

print "Start training >>"
print "node_num = "+str(node_num)+", num_of_train = "+str(num_of_train)+", batch_size = "+str(batch_size)
for loop in range(0,num_of_train):
    print "train loop = ", loop+1
    train_batch = train_data.filter(lambda (data,index): index%batch_size == loop%batch_size)
    rdd = train_batch.map(lambda (data,index): train(node_num,\
                                           np.expand_dims(data[0],axis=0),\
                                           np.expand_dims(data[1],axis=0),\
                                           syn))
    delta = [0 for _ in range(len(node_num))]
    for i in range(0,len(node_num)-1):
        delta[i] = rdd.map(lambda x: x[i]).mean()
    error = rdd.map(lambda x: x[-1]).mean()

    for i in range(0,len(node_num)-1):
        syn[i] += delta[i]
    
    print error

num_of_test = 10000
succeed = 0

print "Start testing >>"
test_data_array = train_data.takeSample(False, num_of_test)

for loop in range(0,num_of_test):
    test_data = test_data_array[loop][0]
    predict = classify(node_num, np.expand_dims(test_data[0],axis=0), syn)
    if test_data[1][predict] == 1.0:
        succeed = succeed + 1

print "correct : ",succeed
print "wrong : ", (num_of_test - succeed)
print "accurancy : ", succeed*100.0/num_of_test,"%"