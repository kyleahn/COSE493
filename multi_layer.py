import numpy as np
from pyspark import SparkConf, SparkContext

#sigmoid function
#when deriv==true, return the derivative function value instead.
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def func(node_num, X, y, syn):
    l = [[] for _ in range(len(node_num))]
    # l0 is the X
    l[0] = X
    # calculate the value on layer1 & y, with non-linear(sigmoid) function
    for i in range(0,len(node_num)-1):
        l[i+1] = nonlin(np.dot(l[i], syn[i]))

    softmax = [0 for _ in range(node_num[-1])]
    for i in range(0, len(l[-1])):
        z = -np.log(1/l[-1][i]-1)
        softmax[i] = z
    sum_sft = sum(softmax)
    for i in range(0, len(l[-1])):
        softmax[i] = softmax[i] / sum_sft

    error = delta = [0 for _ in range(len(node_num))]
    for i in xrange(len(node_num)-1,0,-1):
        if i == len(node_num)-1:
            error[i] = y - l[i]
        else:
            error[i] = delta[i+1].dot(syn[i].T)
        delta[i] = error[i] * nonlin(l[i], deriv=True)

    ret = []
    for i in range(0,len(node_num)-1):
        ret.append(l[i].T.dot(delta[i+1]))
    ret.append(error[-1])

    return ret

np.random.seed(1)

#init spark
conf = (SparkConf()
        .setAppName("SPARK_ANN")
#        .setMaster("spark://192.168.0.3:7077"))
        .setMaster("local[*]"))
sc = SparkContext(conf=conf)

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
    syn.append(2 * np.random.random((node_num[i], node_num[i+1])) - 1)

for loop in range(1,num_of_train):
    print "train loop = ", loop
    rdd = train_data.map(lambda data: func(node_num,\
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