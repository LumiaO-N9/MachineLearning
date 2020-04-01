import matplotlib.pyplot as plt
from Ch01 import kNN
from numpy import array

datingDataMat, datingLabels = kNN.file2matrix("../data/datingTestData/datingTestSet2.txt")
fig = plt.figure()
# 参数111的意思是：将画布分割成1行1列，图像画在从左到右从上到下的第1块
# ax = fig.add_subplot(331)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax1.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 5.0 * array(datingLabels), 150.0 * array(datingLabels))
ax2.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 5.0 * array(datingLabels), 150.0 * array(datingLabels))
plt.show()
