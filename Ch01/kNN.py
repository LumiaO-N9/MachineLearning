from numpy import *


# 创建数据集，演示一下，之后的group、labels都需要格式化成此格式，后续没有使用
def createDataSet():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])

    labels = ['A', 'A', 'B', 'B']
    return group, labels


# K-近邻算法实现
def classify0(inX, dataSet, labels, k):
    # shape 函数会返回一个tuple，tuple[0]为array大小，tuple[1]为array中每个元素的大小
    # 例如：👆createDataSet函数中的group如果调用shape函数，则返回 (4, 2)
    dataSetSize = dataSet.shape[0]
    # 假设inX为[0,0] tile(inx, (dataSetSize,1))会返回一个大小为dataSize行*1列的array，里面的元素都为inx
    # 目的为了方便后续计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 求平方
    sqDiffMat = diffMat ** 2
    # axis 等于1 表示行相加；axis 等于0 表示列相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开平方根
    distances = sqDistances ** 0.5
    # 返回从小到大排序的索引list
    # 例如：有一个list为[3,4,1,2] 则返回 [2,3,0,1]
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 统计每个类别出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]
