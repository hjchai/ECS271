import numpy as np
from cvxopt import matrix, solvers
import csv

def readTrainData(fileName, trainData=[]):
    with open(fileName, 'r') as train:
        reader = csv.reader(train)
        for row in reader:
            for i in range(len(row)):
                row[i] = int(float(row[i]))
            trainData.append(row)

def readTestData(fileName, testData = []):
    with open(fileName, 'r') as test:
        reader = csv.reader(test)
        for row in reader:
            for i in range(len(row)):
                row[i] = int(float(row[i]))
            testData.append(row)

def kernal_linear(x1, x2):
    return np.dot(x1, x2)

def SVM(trainData, kernal): #trainData has to been pre-processed
    x_data = np.array(trainData)[:,0:-1]
    y_label = np.array(trainData)[:,-1].astype(float)
    sample_size = len(x_data)
    kernal_array = np.zeros((sample_size, sample_size))
    if kernal is "linear":
        for i in range(sample_size):
            for j in range(sample_size):
                kernal_array[i, j] = kernal_linear(x_data[i], x_data[j])

    P = matrix(np.outer(y_label,y_label)*kernal_array)
    q = -1 * matrix(np.ones(sample_size))
    G = -1 * matrix(np.identity(sample_size))
    h = matrix(np.zeros(sample_size))
    A = matrix(np.array(y_label),(1,sample_size))
    b = matrix(0.0)

    opt_result = solvers.qp(P,q,G,h,A,b)
    alphas = np.ravel(opt_result['x'])
    return alphas

def prepare_data(trainData, c):
    trainData_m = np.array(trainData)
    for item in trainData_m:
        if item[-1] == c:
            item[-1] = 1
        else:
            item[-1] = -1
    return trainData_m

def calculate_value(alphas, b, trainData, testData, kernal):
    sample_size = len(trainData)
    test_size = len(testData)
    kernal_array = np.zeros((sample_size, test_size))
    if kernal is "linear":
        for i in range(sample_size):
            for j in range(test_size):
                kernal_array[i, j] = kernal_linear(trainData[i,0:-1], testData[j])
    print alphas
    for i in range(sample_size):
        alphas[i] = alphas[i] * trainData[i,-1]
    print alphas
    print np.dot(alphas, kernal_array)+b
    return np.dot(alphas, kernal_array)+b

def getB(alphas, trainData_modified):
    # Support vectors have non zero lagrange multipliers
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    a = alphas[sv]
    x = trainData_modified[sv,0:-1]
    y = trainData_modified[sv,-1]
    #print "%d support vectors out of %d points" % (len(self.a), n_samples)
    # Intercept
    b = 0

    for i in range(len(a)):
        for j in range(alphas.shape[0]):
            b -= alphas[j] * trainData_modified[j,-1] * kernal_linear(trainData_modified[j,0:-1],x[i])
        b += y[i]
    b /= len(a)
    print b
    return b

if __name__ == "__main__":
    fileName_train = '../pendigits-train.csv'
    fileName_test = '../pendigits-test-nolabels.csv'
    trainData = []
    testData = []
    kernal = 'linear'
    readTrainData(fileName_train,trainData)
    readTestData(fileName_test, testData)
    predictions = []

    decision_value = []

    for i in range(10):
        trainData_modified = prepare_data(trainData,i)
        alphas = SVM(trainData_modified, kernal)
        b = getB(alphas, trainData_modified)
        value = calculate_value(alphas, b, trainData_modified, np.array(testData), kernal)
        decision_value.append(value)

    decision_value_array = np.array(decision_value)
    decision = []
    for j in range(len(testData)):
        tmp = decision_value_array[:,j]
        index = np.argmax(tmp)
        decision.append(index)
    with open('decision_value.txt','w') as outFile:
        row, column = decision_value_array.shape
        for i in range(column):
            for j in range(row):
                outFile.write(str(decision_value_array[j,i])+'\t')
            outFile.write('\n')

    with open('result.txt', 'w') as outfile:
        for item in decision:
            outfile.write(str(item)+'\n')