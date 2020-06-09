from math import log
from Ch02.draw.treePlotter import createPlot


# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']
               ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算指定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 统计dataSet中 各个类别所占比例
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 根据信息熵计算公式计算 dataSet的信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 取轴线前的数据
            # 因为是第一次取，可以直接 = ，否则需要使用extend
            reducedFeatVec = featVec[:axis]
            # 取轴线后的数据
            # 两个list合并需使用exteng 或者 +
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选取数据集划分的最好特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 计算原始数据集的信息熵，以便之后计算信息增益
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 取当前特征列
        featList = [example[i] for example in dataSet]
        # 使用Python集合进行去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / len(dataSet)
            # 根据占比prob求期望
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 求信息增益
        infoGain = baseEntropy - newEntropy
        # 取最大信息增益，并取能获得最大信息增益特征值所在列的索引
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 当所有属性都划分后，某划分后的集合仍存在多个种类
# 则采用如下函数进行投票选举，来确定该集合所属类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 使用sorted函数，借助lambda表达式按字典的value 排序 会返回一个 tuple组成的list，
    # 例如：有一个字典 {'a': 2, 'b': 1, 'c': 3} sorted后返回 [('c', 3), ('a', 2), ('b', 1)]
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


# 构建树
def createTree(dataSet, labels):
    # 取最后一列
    classList = [example[-1] for example in dataSet]
    # 如果当前集合中的所有元素都属于一种类别则可直接确定当前类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果当前数据集只有一列了，即所有特征已被消耗，但数据集的种类还未确定
    # 需使用majorityCnt进行判断
    # 这种情况应该不会出现
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 这里不能直接使用=号，不然会改变原有label
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 通过序列化存储决策树
def storeTree(inputTree, filename="treeStored/classifierStorage.txt"):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


# 通过反序列化读取决策树
def grabTree(filename="treeStored/classifierStorage.txt"):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


# 从实验数据构建并绘制决策树
def createTreeFromData(filename="data/lenses.txt"):
    with open(filename) as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = createTree(lenses, lensesLabels)
        print("直接构建的决策树：")
        print(lensesTree)
        storeTree(lensesTree)
        restoredTree = grabTree()
        print("从文件恢复的决策树：")
        print(restoredTree)
        createPlot(restoredTree)


createTreeFromData()
