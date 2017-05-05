import numpy as np
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
        z = -np.log(1/l[-1][0][i]-1)
        softmax[i] = z
    sum_sft = sum(softmax)
    for i in range(0, node_num[-1]):
        softmax[i] = softmax[i] / sum_sft

    #max of the softmax ==  max of answer
    for i in range(0,node_num[-1]):
        if softmax[i] == max(softmax):
            if y[0][i] == max(y[0]):
                print "succeed"
            else:
                print "failed"
            break

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

np.random.seed(1)

#init spark
conf = (SparkConf()
        .setAppName("SPARK_ANN")
#        .setMaster("spark://192.168.0.3:7077"))
        .setMaster("local[*]"))
sc = SparkContext(conf=conf)

#load from file
path = '/Users/Abj/Downloads/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'

num_of_train = 10

x = y = z = []
result = []
exercise = {
  "Walking" : 0,
  "Jogging" : 1,
  "Upstairs" : 2,
  "Downstairs" : 3,
  "Sitting" : 4,
  "Standing" : 5
}
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

            temp = [0.0 for _ in range(6)]
            temp[exercise.get(line.split(",")[1])] = 1.0
            result.append(temp)

X = zip(x,y,z)
Y = result
train_data = sc.parallelize(zip(zip(x,y,z), Y), 1)

#first = 3, last = 6
node_num = [3,4,4,4,6]

syn = []
for i in range(0,len(node_num)-1):
    syn.append(np.random.random((node_num[i], node_num[i+1])))

print "Start training >>"
for loop in range(1,num_of_train):
    print "train loop = ", loop
    rdd = train_data.map(lambda data: train(node_num,\
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