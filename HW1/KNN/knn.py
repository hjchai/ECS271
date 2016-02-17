__author__ = 'huajun'
import csv
import math
import heapq
import numpy as np

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

def knn(testCase, trainData, k):
    dist = []
    for item in trainData:
        tmp = 0
        for i in range(len(testCase)):
            tmp += (item[i]-testCase[i])**2
        dist.append([math.sqrt(tmp),item[-1]])
    k_prediction_index = np.argpartition(np.array(dist)[:,0], k)[:k].tolist()
    k_prediction = []
    for index in k_prediction_index:
        k_prediction.append(dist[index][1])
    prediction_map = {}
    max_count = ('',0)
    for pre in k_prediction:
        if pre in prediction_map:
            prediction_map[pre] += 1
        else:
            prediction_map[pre] = 1
        if prediction_map[pre] > max_count[1]:
            max_count = (pre, prediction_map[pre])
    return max_count[0]

if __name__ == '__main__':
    fileName_train = '../pendigits-train.csv'
    fileName_test = '../pendigits-test-nolabels.csv'
    trainData = []
    testData = []
    readTrainData(fileName_train,trainData)
    readTestData(fileName_test, testData)
    predictions = []
    for i in range(len(testData)):
        prediction = knn(testData[i],trainData,k=6)
        predictions.append(prediction)
    print predictions