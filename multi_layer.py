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

    error = [0 for _ in range(len(node_num))]
    for i in xrange(len(node_num)-1,-1,-1):
        if i == len(node_num)-1:
            error[i] = y - softmax
        else:
            error[i] = error[i+1].dot(syn[i].T) * nonlin(l[i], deriv=True)

    gradient = []
    for i in range(0,len(node_num)-1):
        gradient.append(l[i].T.dot(error[i+1]))
    gradient.append(error[-1])

    return gradient

def classify(node_num, X, syn):
    def nonlin(x, deriv=False):
        if (deriv == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    l = [[] for _ in range(len(node_num))]
    l[0] = X
    for i in range(0,len(node_num)-1):
        l[i+1] = nonlin(np.dot(l[i], syn[i]))
#        print "l[",i,"] = ",l[i]
#        print "syn[",i,"] = ",syn[i]
    return l[-1][0]

if __name__ == "__main__":
#    np.random.seed(1)

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

    print "<<Preprocessing>>"
    csv = sc.textFile(path,24)
    #filter weird data
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

    #drop outlier
    train_data = train_data.filter(lambda (data, index): -100<data[0]<100 and -100<data[1]<100 and -100<data[2]<100)

    #50,000 row in each exercise
    print "modifying rdd with 50,000 row in each exercise"
    temp_train_data = sc.parallelize([])
    for i in range(0,6):
        temp_rdd = train_data.filter(lambda data: data[1][i] == 1.0)
        temp_rdd = temp_rdd.zipWithIndex()
        temp_count = temp_rdd.count()
        temp_rdd = temp_rdd.filter(lambda (data, index) : index%(temp_count/50000) == 0)
        temp_rdd = temp_rdd.map(lambda (data, index) : data)
        temp_train_data = temp_train_data.union(temp_rdd)

    train_data = temp_train_data
    print train_data.first()
    #shuffle rdd
    train_data = train_data.repartition(24)
    print train_data.first()


    #normalization to 0~1
    x_rdd = train_data.map(lambda (data, index): data[0])
    x_max = x_rdd.max(); x_min = x_rdd.min()
    y_rdd = train_data.map(lambda (data, index): data[1])
    y_max = y_rdd.max(); y_min = y_rdd.min()
    z_rdd = train_data.map(lambda (data, index): data[2])
    z_max = z_rdd.max(); z_min = z_rdd.min()

    print x_max, x_min
    print y_max, y_min
    print z_max, z_min

    normalization_factor = 10000.0
    print "normalization to +-",normalization_factor
    train_data = train_data.map(lambda (data, index):
                                (( (data[0]-x_min-(x_max-x_min)/2.0)/(x_max-x_min)*normalization_factor,
                                   (data[1]-y_min-(y_max-y_min)/2.0)/(y_max-y_min)*normalization_factor,
                                   (data[2]-z_min-(z_max-z_min)/2.0)/(z_max-z_min)*normalization_factor ), index) )


    #first = 3, last = 6
    node_num = [3,500,500,6]
    num_of_train = 10
    batch_size = 1

    syn = []
    for i in range(0,len(node_num)-1):
        # rand in -0.1 ~ +0.1
        syn.append(np.random.random((node_num[i], node_num[i+1]))*0.2-0.1)

    train_data = train_data.zipWithIndex()
    print train_data.first()
    train_data = train_data.repartition(24)
#    print train_data.collect()

    #split test and train data
    test_ratio = 1.0
    rdd_size = train_data.count()

    test_data = train_data.filter(lambda (data,index): index%int((100/test_ratio)) == 0)
    train_data = train_data.filter(lambda (data,index):  index%int((100/test_ratio)) != 0)

   # print train_data.collect()

    print "analyzing train_data..."
    for i in range(6):
        print i," count : ", train_data.filter(lambda (data,index): data[1][i] == 1.0).count()
    print "analyzing test_data..."
    for i in range(6):
        print i," count : ", test_data.filter(lambda (data,index): data[1][i] == 1.0).count()

    print "Start training >>"
    print "node_num = "+str(node_num)+", num_of_train = "+str(num_of_train)+", batch_size = "+str(batch_size)
    for loop in range(0,num_of_train):
        print "train loop = ", loop+1
        train_batch = train_data.filter(lambda (data,index): index%batch_size == loop%batch_size)

        print "["
        for i in range(6):
            print train_batch.map(lambda (data, index): data[1][i]).sum(), ","
        print "]"

        rdd = train_batch.map(lambda (data,index): train(node_num,\
                                               np.expand_dims(data[0],axis=0),\
                                               np.expand_dims(data[1],axis=0),\
                                               syn))
        delta = [0 for _ in range(len(node_num))]
        for i in range(0,len(node_num)-1):
            delta[i] = rdd.map(lambda x: x[i]).mean()
        error = rdd.map(lambda x: x[-1]).mean()

        #alpha if learning rate
        alpha = 0.05
        for i in range(0,len(node_num)-1):
            syn[i] += alpha * delta[i]

        print error

    num_of_test = test_data.count()

    print "Start testing >>"
    print "num_of_test = "+str(num_of_test)

    test_data = test_data.map(lambda (data, index): (data[0], data[1], classify(node_num, np.expand_dims(data[0],axis=0), syn)))
    print test_data.map(lambda data: (data[0], data[1], data[2])).collect()

    test_result = test_data.filter(lambda data: data[1][data[2].tolist().index(max(data[2]))] == 1.0)
    succeed = test_result.count()

#    print "test"
#    for i in range(11):
#        print "[", i / 10.0 * normalization_factor, ",", i / 10.0 * normalization_factor, ",", i / 10.0 * normalization_factor, "] -> ", classify(
#            node_num, np.expand_dims(
#                [i / 10.0 * normalization_factor, i / 10.0 * normalization_factor, i / 10.0 * normalization_factor],
#                axis=0), syn)

    print "correct : ",succeed
    print "wrong : ", (num_of_test - succeed)
    print "accurancy : ", succeed*100.0/num_of_test,"%"
