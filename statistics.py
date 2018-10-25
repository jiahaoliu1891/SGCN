import numpy as np
from BPdata import *
from matplotlib import pyplot

if __name__ == "__main__":
    dataSet = BPDataSet()
    data = dataSet.getData()
    graphSize = dataSet.getGraphSize()

    DTI, fMRI, label = data["DTI"], data["fMRI"], data["label"]

    testIDX = 0

    DTI_degree = DTI.copy()
    # DTI_degree[DTI_degree > 0] = 1
    DTI_degree = np.sum(DTI_degree, axis=1)

    # 画出度随着节点的分布
    pyplot.hist(DTI_degree[2], 50)
    pyplot.show()
    for i in range(graphSize):
        print(DTI_degree[i].argmax())



