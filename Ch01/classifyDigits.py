from os import listdir

from numpy import *
from Ch01.kNN import classify0


# 图片都是32*32的，每个像素都用0或1表示
def img2vector(filename):
    returnVect = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 列出目录下所有文件
    trainingDataPath = 'data/digitsTestData/digits/trainingDigits'
    testDataPath = 'data/digitsTestData/digits/testDigits'
    trainingFileList = listdir(trainingDataPath)
    print(trainingFileList)
    filesCount = len(trainingFileList)
    trainingMat = zeros((filesCount, 1024))
    for i in range(filesCount):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(trainingDataPath + '/' + fileNameStr)

    testFileList = listdir(testDataPath)
    errorCount = 0.0
    testFilesCount = len(testFileList)
    for i in range(testFilesCount):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        testMat = img2vector(testDataPath + '/' + fileNameStr)
        # 不是1就是0，所以不需要做归一化处理
        classifierResult = int(classify0(testMat, trainingMat, hwLabels, 3))
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is : %f" % (errorCount / testFilesCount))


handwritingClassTest()
