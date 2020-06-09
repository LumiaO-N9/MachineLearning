from numpy import *
from Ch01.kNN import classify0


# 把文件转换成矩阵
def file2matrix(filename):
    with open(filename) as fr:
        arrayOfLines = fr.readlines()
        numberOfLines = len(arrayOfLines)
        # zeros 生成numberOfLines行*3列，值都为0的矩阵
        returnMat = zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split('\t')
            # "index,"表示一个tuple，代表一个array的第几行第几行
            # 这在Python的list中是不支持的
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(listFromLine[-1])
            index += 1
        return returnMat, classLabelVector


# 归一化数值
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - tile(minVals, (m, 1))) / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLables = file2matrix("data/datingTestData/datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLables[numTestVecs:m], 3)
        print("the classified came back with: %s, the real answer is : %s" % (classifierResult, datingLables[i]))
        if (classifierResult != datingLables[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('data/datingTestData/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[int(classifierResult) - 1])


classifyPerson()
