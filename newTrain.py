# coding=utf-8

import torch
from torch import optim
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
from BPdata import BPDataSet
from model import *
import scipy.io as sio


def compute_acc(net, testData):
    testfMRI, testDTI, testLabel, testSize = testData
    X = net.test(testfMRI, testDTI).detach().numpy()
    X[X >= 0.5] = 1
    X[X < 0.5] = 0
    return ((X == testLabel).sum()) / testSize


def get_train_shuffle(trainfMRI, trainDTI, trainLabel, trainSize):
    # Siamese 网络用的 shuffle 数据
    idx = np.arange(trainSize)
    np.random.shuffle(idx)
    # 注意这里需要使用.clone() 保证是原始数据和进行shuffle的数据是两个不同的内存id, 不然原始数据也会变乱
    trainfMRI_S = trainfMRI.clone()[idx]
    trainDTI_S = trainDTI.clone()[idx]
    trainLabel_S = trainLabel.clone()[idx]
    return trainfMRI_S, trainDTI_S, trainLabel_S


def trainSiamese(EPOCH=100, EPOCH_C=100, MARGIN = 10.0, FEAT1 = 32, FEAT2 = 64, LR1 = 0.01, LR2 = 0.005, PLOT = True):

    dataSet = BPDataSet()
    graphSize = dataSet.getGraphSize()
    # print("======================建立网络 Semi-GCN======================")
    net = AddSiameseGCN(graphSize, FEAT1, FEAT2)
    optimizer = optim.Adam(net.parameters(), lr=LR1)
    criteria = AddContrastiveLoss(lamb=1, margin=MARGIN)
    # criteria = torch.nn.BCELoss()

    trainData, testData = dataSet.getDataTensor(trainSize=60)
    trainSize, testSize = dataSet.getTrainSize(), dataSet.getTestSize()
    trainfMRI, trainDTI, trainLabel = trainData['fMRI'], trainData["DTI"], trainData["label"]
    testfMRI, testDTI, testLabel = testData['fMRI'], testData["DTI"], testData["label"]
    testData = (testfMRI, testDTI, testLabel, testSize)

    for epoch in range(EPOCH):
        trainfMRI_S, trainDTI_S, trainLabel_S = get_train_shuffle(trainfMRI, trainDTI, trainLabel, trainSize)
        optimizer.zero_grad()
        siamese_label = (trainLabel == trainLabel_S).float()
        # 输出两个 embedding
        # out1, out2 = net(trainDTI, trainfMRI,  trainDTI_S, trainfMRI_S)
        out1, out2, out_class = net(trainfMRI, trainDTI, trainfMRI_S, trainDTI_S)
        loss = criteria(out1, out2, out_class, siamese_label, trainLabel)
        loss.backward()
        optimizer.step()

        acc = compute_acc(net, testData)

        print("Epoch number {} ".format(epoch))
        print("Current loss {} Accuracy:{}".format(loss.item(), acc))



if __name__ == "__main__":

    import time
    # epoch = 10
    # epoch_C = 90
    # margin = 0.5
    # feat1 = 128
    # feat2 = 128
    # lr1 = 0.01

    hist = []
    epoch = 200
    epoch_C = 45
    margin = 1.5
    feat1 = 256
    feat2 = 128
    lr1 = 0.01
    lr2 = 0.001

    trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, lr2, PLOT=False)

    # print("===========================================参数配置=============================")
    # print("epoch:{}, margin:{}, feat1:{}, feat2:{}, lr1:{}".format(epoch, margin, feat1, feat2, lr1))
    # trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, PLOT=False)
    # for i in range(25):
    #     hist.append(trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, lr2, PLOT=False))
    # print("==================================准确率{}=============================".format(np.mean(hist)))



