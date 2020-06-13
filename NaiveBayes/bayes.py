from numpy import ones, log, array, random, zeros


# 创建一个供测试的单词的集合
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性词汇, 0 相反
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 自定义的一个单词集合
def loadDataSet2():
    postingList = [
        ['a', 'b', 'c', 'd', 'e'],  # 类别为0
        ['b', 'c', 'f', 'g', 'h']  # 类别为1
    ]
    # ['a','d','c','e','h','k']
    classVec = [0, 1]
    return postingList, classVec


# 返回去重后的单词集合
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    # 转成list为了让集合有序
    return list(vocabSet)


# 把words集合转化成向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 训练贝叶斯分类器
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 分子全部置为1
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)  # change to zeros()
    # 分母全部置为2，防止出现 等于0的情况
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    # p0Denom = 0.0
    # p1Denom = 0.0  # change to 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 在假设各个特征相互独立的情况下，我们可以把 概率相乘 通过 ln 转化成加法
    # 因为ln函数是个增函数，我们最后是通过 比较 两个 概率的 大小 来进行 分类
    # 所以 这种转化不会 影响结果
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 在 setOfWords2Vec中，我们只考虑了单词有没有出现，
# 并没有考虑单词出现的次数
# 但通常 单词频率出现的越高
# 其 对结果的影响越大
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 使用正则表达式 过滤 不需要的单词
# 当然实际情况中，需要使用停词表
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 垃圾邮件测试
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('data/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = list(range(50))
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


# 计算出现次数最高的前30个单词
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 排序
    sortedFreq = sorted(freqDict.items(), key=lambda x: x[1], reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['title_detail']['value'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['title_detail']['value'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(milwaukee, la_lakers):
    vocabList, p0V, p1V = localWords(milwaukee, la_lakers)
    topMilwaukee = []
    topLakers = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topLakers.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topMilwaukee.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topLakers, key=lambda pair: pair[1], reverse=True)
    print("******************LA-Lakers***************************************")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topMilwaukee, key=lambda pair: pair[1], reverse=True)
    print("*******************Milwaukee**************************************")
    for item in sortedNY:
        print(item[0])


# 从RSS源获取数据 并测试我们的贝叶斯分类器
# 最后按频率列出 两个球队新闻的常用词
def testBayesFromRSS():
    import feedparser
    # 从Yahoo的RSS源上获取有关雄鹿队的news
    milwaukee = feedparser.parse('https://sports.yahoo.com/nba/teams/milwaukee/rss.xml')
    # 从Yahoo的RSS源上获取有关湖人队的news
    la_lakers = feedparser.parse('https://sports.yahoo.com/nba/teams/la-lakers/rss.xml')
    for i in range(10):
        vocabList, pXL, pHR = localWords(milwaukee, la_lakers)
    getTopWords(milwaukee, la_lakers)


def testMyOwnSet():
    # a 代表输入数据集，b 代表 类别
    a, b = loadDataSet2()
    c = ['a','d','c','e','h','k']
    my = createVocabList(a)
    trainMat = []
    for i in a:
        trainMat.append(setOfWords2Vec(my, i))
    p0, p1, pAb = trainNB0(trainMat, b)
    print('对C分类的结果为', classifyNB(array(setOfWords2Vec(my,c)), p0, p1, pAb))


if __name__ == '__main__':
    testMyOwnSet()
    # testBayesFromRSS()
