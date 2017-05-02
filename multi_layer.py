import numpy as np
from pyspark import SparkConf, SparkContext

#sigmoid function
#when deriv==true, return the derivative function value instead.
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def func(node_num, X, y, syn):

    l = []
    # l0 is the X
    l[0] = X
    # calculate the value on layer1 & y, with non-linear(sigmoid) function
    for i in range(0,len(node_num)-1):
        l[i+1] = nonlin(np.dot(l[i], syn[i]))

    error = delta = []
    for i in xrange(len(node_num)-1,0,-1):
        if i == len(node_num)-1:
            error[i] = y - l[i]
        else:
            error[i] = delta[i+1].dot(syn[i].T)
        delta[i] = error[i] * nonlin(l[i], deriv=True)

    temp = []
    for i in range(0,len(node_num)-1):
        temp.append(l[i].T.dot(delta[i+1]))
    temp.append(error[len(node_num-1)])

    return temp


np.random.seed(1)

#init spark
conf = (SparkConf()
        .setAppName("SPARK_ANN")
#        .setMaster("spark://192.168.0.3:7077"))
        .setMaster("local[*]"))
sc = SparkContext(conf=conf)

#broadcasting the syn0, syn1 allow the accessing these variables from other clusters.

#load from file
path = '/home/master/Downloads/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'

#preprocess file
with open(path, 'r') as file :
    filedata = file.read()
filedata = filedata.replace("Walking", "0")
filedata = filedata.replace("Jogging", "1")
filedata = filedata.replace("Upstairs", "2")
filedata = filedata.replace("Downstairs", "3")
filedata = filedata.replace("Sitting", "4")
filedata = filedata.replace("Standing", "5")
with open(path, 'w') as file :
    file.write(filedata)

#read file
#rdd = sc.textFile(path) \
#    .filter(lambda line: len(line.split(","))>=6) \
#    .map(lambda line: (line.split(",")[0],line.split(",")[1],line.split(",")[2],line.split(",")[3],line.split(",")[4],line.split(",")[5])) \
#    .collect()

num_of_train = 10

x = y = z = []
result = []
with open(path, 'r') as file :
    while True:
        line = file.readline()
        if not line: break
        if len(line.split(","))>=6 \
                and len(line.split(",")[3])>0 \
                and len(line.split(",")[4])>0 \
                and len(line.split(",")[5])>1:
            x.append(float(line.split(",")[3]))
            y.append(float(line.split(",")[4]))
            z.append(float(line.split(",")[5].replace(";","")))
            result.append(int(line.split(",")[1]))

X = zip(x,y,z)
Y = np.expand_dims(result, axis=1)
train_data = sc.parallelize(zip(zip(x,y,z), Y))

#first = 3, last = 6
node_num = [3,4,4,4,6]

syn = []
for i in range(0,len(node_num)-1):
    syn[i] = 2 * np.random.random((node_num[i], node_num[i+1])) - 1

for loop in range(1,num_of_train):
    print "train loop = ", loop
    rdd = train_data.map(lambda data: func(node_num,\
                                           np.expand_dims(data[0],axis=0),\
                                           np.expand_dims(data[1],axis=0),\
                                           syn))
    delta = []
    for i in range(0,len(node_num)-1):
        delta[i] = rdd.map(lambda x: x[i]).mean()
    error = rdd.map(lambda x: x[-1]).mean()

    for i in range(0,len(node_num)-1):
        syn[i] += delta[i]
    
    print error




'''
#(temp)
import numpy as np
from pyspark import SparkConf, SparkContext

#sigmoid function
#when deriv==true, return the derivative function value instead.
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def func(idx, syn0, syn1):
    # input sample
    orig_X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # y data
    orig_y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    X = np.expand_dims(orig_X[idx], axis=0)
    y = np.expand_dims(orig_y[idx], axis=0)
    
    # l0 is the X
    l0 = X
    # calculate the value on layer1 & y, with non-linear(sigmoid) function
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # calculate the error by just subtracting
    # i think it can be change to other method like MSE, cross entropy...etc
    l2_error = y - l2

    # logging
    print "loop >", "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # refresh the syn0, syn1 value
    # by adding the l1 dot l2(delta)
    
    return (l1.T.dot(l2_delta), l0.T.dot(l1_delta), l2_error)

np.random.seed(1)

#syn0 is weight that from X to layer1
syn0 = 2 * np.random.random((3, 4)) - 1
#syn1 is weight that from layer1 to y
syn1 = 2 * np.random.random((4, 1)) - 1

#init spark
conf = (SparkConf()
        .setAppName("SPARK_ANN")
        .setMaster("spark://192.168.0.3:7077"))
#        .setMaster("local[*]"))
sc = SparkContext(conf=conf)

#broadcasting the syn0, syn1 allow the accessing these variables from other clusters.

num_of_train = 120

for loop in range(1,num_of_train):
    print "train loop = ", loop
    pa = sc.parallelize(range(0,3))
    rdd = pa.map(lambda idx: func(idx,syn0,syn1))
    rdd.collect()
    
    delta_l1 = rdd.map(lambda x: x[0]).mean()
    delta_l0 = rdd.map(lambda x: x[1]).mean()
    error = rdd.map(lambda x: x[2]).mean()

    syn1 += delta_l1
    syn0 += delta_l0
    
    print error
'''