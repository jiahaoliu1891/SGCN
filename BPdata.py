# coding=utf-8

import scipy.io as sio
import torch
import numpy as np
from utils import *
norm = lambda x: (x-x.min())/(x.max()-x.min())


def normalize(data):
    D = np.diag(data.copy())
    a = np.arange(data.shape[-1])
    data[a, a] = 0
    data = norm(data)
    data[a, a] = norm(D)
    return data


class BPDataSet(object):
    def __init__(self):
        DTI_dataSet = sio.loadmat("./data/BP_DTI.mat")
        DTI_obj = DTI_dataSet['data'][:, 0]
        DTI_matrix = self.obj2array(DTI_obj)

        a = np.arange(DTI_matrix.shape[-1])
        if DTI_matrix.max() > 1:
            for i in range(DTI_matrix.shape[0]):
                DTI_matrix[i][a, a] = 0
                DTI_matrix[i] = norm(DTI_matrix[i])

        label = DTI_dataSet['label']
        label[label == -1] = 0

        fMRI_dataSet = sio.loadmat("./data/BP_fMRI.mat")
        fMRI_obj = fMRI_dataSet['data'][:, 0]
        fMRI_matrix = self.obj2array(fMRI_obj)

        self.dataSize = label.shape[0]
        self.graphSize = fMRI_matrix[0, 0].shape[0]
        self.trainSize, self.testSize = None, None
        self.idx = None
        self.data = {
            "fMRI": fMRI_matrix,
            "DTI": DTI_matrix,
            "label": label
        }

    def getDataNP(self, trainSize):
        self.trainSize = trainSize
        self.testSize = self.dataSize - trainSize

        # self.idx = np.arange(self.dataSize)
        # np.random.shuffle(self.idx)

        DTI = self.data["DTI"].copy()
        fMRI = self.data['fMRI'].copy()
        label = self.data["label"].copy()

        trainDTI, trainfMRI, trainLabel = map(lambda x: x[self.idx][:trainSize], [DTI, fMRI, label])
        testDTI, testfMRI, testLabel = map(lambda x: x[self.idx][trainSize:], [DTI, fMRI, label])

        self.trainData = {
            "DTI": trainDTI,
            "fMRI": trainfMRI,
            "label": trainLabel
        }
        self.testData = {
            "DTI": testDTI,
            "fMRI": testfMRI,
            "label": testLabel
        }
        return self.trainData, self.testData

    def getDataTensor(self, trainSize):
        trainData, testData = self.getDataNP(trainSize)

        wrapTensor = lambda data: torch.Tensor(data)
        trainData["fMRI"], testData["fMRI"], trainData["DTI"], testData["DTI"], trainData["label"] = \
            map(wrapTensor, [trainData["fMRI"], testData["fMRI"], trainData["DTI"], testData["DTI"], trainData["label"]])

        return trainData, testData

    def makeNewData(self):
        np.save("./log/idx.npy", self.idx)

    def normalize(self, data):
        normData = np.zeros_like(data)
        for i in range(data.shape[0]):
            min = data[i].min()
            max = data[i].max()
            normData[i] = (data[i] - min)/(max - min)
        return normData

    def obj2array(self, data):
        length = data.shape[0]
        size = data[0].shape[0]
        arrayData = np.zeros((length, size, size))
        for i in range(data.shape[0]):
            arrayData[i] = data[i]
        return arrayData

    def GenDTICSV(self):
        for i in range(self.dataSize):
            with open("./CSV/edge_DTI{}.csv".format(i), "w") as f:
                f.write("Id,Target,Source,Weight,Type\n")
                W = self.data["DTI"][i]
                count = 0
                for k in range(self.graphSize):
                    for l in range(k+1, self.graphSize):
                        f.write("{},{},{},{},{}\n".format(count, k, l, W[k, l], "Undirected"))
                        count += 1

    def GenfMRICSV(self):
        for i in range(self.dataSize):
            with open("./CSV/edge_fMRI{}.csv".format(i), "w") as f:
                f.write("Id,Target,Source,Weight,Type\n")
                W = self.data["fMRI"][i]
                count = 0
                for k in range(self.graphSize):
                    for l in range(k+1, self.graphSize):
                        f.write("{},{},{},{},{}\n".format(count, k, l, W[k, l], "Undirected"))
                        count += 1

    def play(self):
        DTI = self.data["DTI"].copy()
        fMRI = self.data['fMRI'].copy()
        label = self.data["label"].copy()

        for i in range(self.dataSize):
            print(np.sum(DTI[i], axis=0).min())
            print(np.sum(fMRI[i], axis=0).min())

    def DAD(self):
        pass

    def getTrainSize(self):
        return self.trainSize

    def getGraphSize(self):
        return self.graphSize

    def getDataSize(self):
        return self.dataSize

    def getData(self):
        return self.data

    def getTestSize(self):
        return self.testSize

    def getIdx(self):
        return self.idx


if __name__ == "__main__":
    a = BPDataSet()
    data = a.getData()
    fMRI = data['fMRI']
    DTI = data['DTI']

    # 统计度
    for k in range(DTI.shape[0]):
        degree = compyte_degree(DTI[k])
        plt.hist(degree, 50)
        plt.show()
        for i in range(degree.shape[0]):
            print("i:{},    degree:{}".format(i, degree[i]))




