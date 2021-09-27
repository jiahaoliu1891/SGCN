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
    idx = np.arange(trainSize)
    np.random.shuffle(idx)
    trainfMRI_S = trainfMRI.clone()[idx]
    trainDTI_S = trainDTI.clone()[idx]
    trainLabel_S = trainLabel.clone()[idx]
    return trainfMRI_S, trainDTI_S, trainLabel_S


def trainSiamese(EPOCH=100, EPOCH_C=100, MARGIN = 10.0, FEAT1 = 32, FEAT2 = 64, LR1 = 0.01, LR2 = 0.005, PLOT = True):

    dataSet = BPDataSet()
    graphSize = dataSet.getGraphSize()
    net = AddSiameseGCN(graphSize, FEAT1, FEAT2)
    optimizer = optim.Adam(net.parameters(), lr=LR1)
    criteria = AddContrastiveLoss(lamb=1, margin=MARGIN)

    trainData, testData = dataSet.getDataTensor(trainSize=60)
    trainSize, testSize = dataSet.getTrainSize(), dataSet.getTestSize()
    trainfMRI, trainDTI, trainLabel = trainData['fMRI'], trainData["DTI"], trainData["label"]
    testfMRI, testDTI, testLabel = testData['fMRI'], testData["DTI"], testData["label"]
    testData = (testfMRI, testDTI, testLabel, testSize)

    for epoch in range(EPOCH):
        trainfMRI_S, trainDTI_S, trainLabel_S = get_train_shuffle(trainfMRI, trainDTI, trainLabel, trainSize)
        optimizer.zero_grad()
        siamese_label = (trainLabel == trainLabel_S).float()
        out1, out2, out_class = net(trainfMRI, trainDTI, trainfMRI_S, trainDTI_S)
        loss = criteria(out1, out2, out_class, siamese_label, trainLabel)
        loss.backward()
        optimizer.step()

        acc = compute_acc(net, testData)

        print("Epoch number {} ".format(epoch))
        print("Current loss {} Accuracy:{}".format(loss.item(), acc))



if __name__ == "__main__":

    import time
   
    hist = []
    epoch = 200
    epoch_C = 45
    margin = 1.5
    feat1 = 256
    feat2 = 128
    lr1 = 0.01
    lr2 = 0.001

    trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, lr2, PLOT=False)
