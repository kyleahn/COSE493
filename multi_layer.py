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
    return l[-1][0]

if __name__ == "__main__":

    #init spark
    conf = (SparkConf()
            .setAppName("SPARK_ANN")
            .setMaster("spark://192.168.0.3:7077"))
    #        .setMaster("local[*]"))
    sc = SparkContext(conf=conf)

    #load from file
    if platform.system() == 'Linux':
        path = '/home/master/Downloads/WISDM_ar_v1.1/WISDM_spectrum_40_overlap_20_train.csv'
    elif platform.system() == 'Windows':
        path = 'C:\Users\KUsch\Downloads\WISDM_at_v2.0\WISDM_at_v2.0_raw.txt'
    else:
        path = '/Users/Abj/Downloads/WISDM_at_v2.0/WISDM_at_v2.0_raw.txt'

    print "<<Preprocessing>>"
    csv = sc.textFile(path,24)
    #filter weird data
    data = csv.map(lambda line: (line.split(",")))
    #form change to (X,Y,Z), result
    def change(line):
        for i in range(63):
            line[i] = float(line[i])
        line[63] = int(round(float(line[63])))-1
        temp = [0.0 for _ in range(6)]
        temp[line[63]] = 1.0
        temp_list = [line[i] for i in range(63)]
        return (tuple(temp_list), temp)
    train_data = data.map(lambda line: change(line))

    #first = 3, last = 6
    node_num = [63,6]
    num_of_train = 100
    batch_size = 3000

    syn = []
    for i in range(0,len(node_num)-1):
        # rand in -0.1 ~ +0.1
        syn.append(np.random.random((node_num[i], node_num[i+1]))*0.2-0.1)

    train_data = train_data.zipWithIndex()

    #split test and train data
    test_ratio = 10.0
    rdd_size = train_data.count()

    test_data = train_data.filter(lambda (data,index): index%int((100/test_ratio)) == 0)
    train_data = train_data.filter(lambda (data,index):  index%int((100/test_ratio)) != 0)

    accurancy = [0.0]
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

        #alpha if learning rate
        alpha = 0.01
        for i in range(0,len(node_num)-1):
            syn[i] += alpha * delta[i]

        num_of_test = test_data.count()

        print "Start testing >>"
        print "num_of_test = " + str(num_of_test)

        test_data_rdd = test_data.map(lambda (data, index): (data[0], data[1], classify(node_num, np.expand_dims(data[0], axis=0), syn)))

        test_result = test_data_rdd.filter(lambda data: data[1][data[2].tolist().index(max(data[2]))] == 1.0)
        succeed = test_result.count()

#        print "correct : ", succeed
#        print "wrong : ", (num_of_test - succeed)
        print "accurancy : ", succeed * 100.0 / num_of_test, "%"

        accurancy.append(succeed * 100.0 / num_of_test)

    import matplotlib.pyplot as plt
    plt.title('change in accurancy at node='+str(node_num))
  #  plt.axes([1,num_of_train,0.0,100.0])
    plt.plot(accurancy)
    plt.xlabel('loop')
    plt.ylabel('accurancy')
    plt.show()

