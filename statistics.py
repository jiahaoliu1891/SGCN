import numpy as np
from BPdata import *
from matplotlib import pyplot
from utils import *
if __name__ == "__main__":
    dataSet = BPDataSet()
    data = dataSet.getData()
    graphSize = dataSet.getGraphSize()

    a = np.arange(graphSize)
    DTI, fMRI, label = data["DTI"], data["fMRI"], data["label"]

    testIDX = 0


    for k in range(10):
        DTI_degree = DTI[k].copy()
        DTI_degree = np.sum(DTI_degree, axis=1)

        # 画出度随着节点的分布
        # pyplot.hist(DTI_degree, 50)
        # pyplot.show()

        print(nsmall_index(DTI_degree, 10))