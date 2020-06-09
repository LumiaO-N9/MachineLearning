from matplotlib import pyplot as plt
# 解决中文乱码问题
from matplotlib.font_manager import FontProperties

# 使用matplotlib的文本注解来绘制树形图

decisionNode = dict(boxstyle="sawtooth", fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 获取系统自带字体 MacOS、Windows的字体路径不同
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size=15)


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def plotNodeCH(nodeTxt, centerPt, parentPt, nodeType, myFont):
    createPlot.ax1.annotate(nodeTxt, fontproperties=myFont, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    # 直接使用中文会出现乱码问题，因为matplotlib库原生不支持显示中文
    plotNode('decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def createPlotCH():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    # 获取自定义字体
    myFont = getChineseFont()
    # 使用系统字体，绘制带中文的注解
    plotNodeCH('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode, myFont)
    plotNodeCH('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode, myFont)
    plt.show()


# 获取叶子结点的个数，以便确定x轴的长度
def getNumLeafs(myTree):
    numLeafs = 0
    # 根节点是唯一的，所以可以直接使用keys()[0]获取
    # 其他情况下dict是无序的
    firstStr = list(myTree.keys())[0]
    secondDict: dict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# getNumLeafs 与 getTreeDepth 合并
def getNumLeafsAndDepth(myTree: dict):
    numLeafs = 0
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            numLeafsTemp, maxDepthTemp = getNumLeafsAndDepth(secondDict[key])
            numLeafs += numLeafsTemp
            thisDepth = 1 + maxDepthTemp
        else:
            numLeafs += 1
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return numLeafs, maxDepth


# 存储两棵树 避免每次都使用 trees.py中的createTree来创建 方便我们测试绘图
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


# 在 父子节点 的连线 中间 填充文本
def plotMidText(cntrPt, parentPt, txtString):
    # 计算带填充文本的x轴坐标
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    # 计算带填充文本的y轴坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs, depth = getNumLeafsAndDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 创建绘图区，并调用 plotTree函数绘制决策树
# 只能绘制英文 若要绘制中文 可参考之前的createPlotCH()函数
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW, plotTree.totalD = getNumLeafsAndDepth(inTree)
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
