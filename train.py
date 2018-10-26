# coding=utf-8

import torch
from torch import optim
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
from BPdata import BPDataSet
from model import *
from config import *

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

    # EPOCH = 90
    # margin = 4 的时候, classnet 要训练好久（>1000）才能收敛， 等于10 的时候 200多epoch久收敛了
    # MARGIN = 10
    # FEAT1 = 32
    # FEAT2 = 64
    # LR1 = 0.01
    # PLOT = False

    dataSet = BPDataSet()
    graphSize = dataSet.getGraphSize()
    # print("======================建立网络 Semi-GCN======================")
    net = SiameseGCN(graphSize, FEAT1, FEAT2)
    optimizer = optim.Adam(net.parameters(), lr=LR1)
    criteria = ContrastiveLoss(margin=MARGIN)
    # criteria = torch.nn.BCELoss()

    trainData, testData = dataSet.getDataTensor(trainSize=60)

    trainSize, testSize = dataSet.getTrainSize(), dataSet.getTestSize()
    trainfMRI, trainDTI, trainLabel = trainData['fMRI'], trainData["DTI"], trainData["label"]
    testfMRI, testDTI, testLabel = testData['fMRI'], testData["DTI"], testData["label"]

    if torch.cuda.is_available():
        # print("CUDA is available!")
        net.cuda()
        trainfMRI, trainDTI, testfMRI, testDTI, trainLabel = map(lambda x: x.cuda(), [trainfMRI, trainDTI, testfMRI, testDTI, trainLabel])

    I = torch.Tensor(np.eye(graphSize))

    for epoch in range(EPOCH):
        trainfMRI_S, trainDTI_S, trainLabel_S = get_train_shuffle(trainfMRI, trainDTI, trainLabel, trainSize)
        optimizer.zero_grad()
        siamese_label = (trainLabel == trainLabel_S).float()
        # 输出两个 embedding
        # out1, out2 = net(trainDTI, trainfMRI,  trainDTI_S, trainfMRI_S)
        out1, out2 = net(trainfMRI, trainDTI, trainfMRI_S, trainDTI_S)
        loss = criteria(out1, out2, siamese_label)
        loss.backward()
        optimizer.step()

        # print("Epoch number {} ".format(epoch))
        # print("Current loss {} Accuracy:{}".format(loss.item(), 0))

        if (epoch % 50 == 0 or epoch == EPOCH-1) and PLOT:
            # 先将 训练数据 和 测试数据 合并起来， 然后进行一起使用 t-SNE 进行降维
            W_vis = torch.cat((trainfMRI, testfMRI), dim=0)
            F_vis = torch.cat((trainDTI, testDTI), dim=0)
            y1 = trainLabel.detach().numpy().reshape(trainSize)
            y2 = testLabel.reshape(testSize)
            y = np.hstack((y1, y2))
            # 求出所有数据的 embedding
            X = net.forward_once(W_vis, F_vis)
            X = X.detach().numpy()
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, perplexity=30)
            X_tsne = tsne.fit_transform(X)
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

            # 画 train data 的分布
            plt.figure(figsize=(6, 6))
            plt.scatter(X_norm[:trainSize, 0], X_norm[:trainSize, 1], s=150, c=y[:trainSize], alpha=.7)
            plt.xticks([])
            plt.yticks([])
            plt.title("train {}".format(epoch))
            plt.show()

            plt.scatter(X_norm[trainSize:, 0], X_norm[trainSize:, 1], s=150, marker="^", c=y[trainSize:], alpha=.7)
            plt.xticks([])
            plt.yticks([])
            plt.title("test {}".format(epoch))
            plt.show()


    embeddingsTrain = net.forwardDownSample(trainfMRI, trainDTI).detach()
    embeddingsTest = net.forwardDownSample(testfMRI, testDTI).detach()
    classNet = ClassifyNetwork((graphSize - NSMALL) * FEAT2)

    if torch.cuda.is_available():
        # print("CUDA is available!")
        classNet.cuda()
        embeddingsTrain, embeddingsTest =  embeddingsTrain.cuda(), embeddingsTest.cuda()

    acc_list = trainClassNet(classNet, embeddingsTrain, embeddingsTest, trainLabel, testLabel, testSize, EPOCH_C, LR2)

    return np.array(acc_list)


def trainClassNet(classNet, embeddingsTrain, embeddingsTest, trainLabel, testLabel, testSize, EPOCH_C, LR2):
    train_size = trainLabel.size(0)
    optm = optim.Adam(classNet.parameters(), lr=LR2)

    def compute_acc():
        X = classNet(embeddingsTest).cpu().detach().numpy() if torch.cuda.is_available() else classNet(embeddingsTest).detach().numpy()
        X[X >= 0.5] = 1
        X[X < 0.5] = 0
        return ((X == testLabel).sum()) / testSize

    loss_function = nn.BCELoss()
    idx = np.arange(train_size)
    acc = 0

    acc_list = []

    for i in range(EPOCH_C):
        np.random.shuffle(idx)
        embeddings_shuffle = embeddingsTrain[idx]
        train_label_shuffle = trainLabel[idx]
        optm.zero_grad()

        prob = classNet.forward(embeddings_shuffle)

        Loss = loss_function(prob, train_label_shuffle)
        Loss.backward()
        optm.step()

        acc = compute_acc()
        # print("Epoch number {} ".format(i))
        # print("Current loss {} Accuracy:{}".format(Loss.item(), acc))

        if i in range(0, 500):
            acc_list.append(acc)

    return acc_list


if __name__ == "__main__":

    import time
    hist = np.zeros((25, 500))
    epoch_list = range(7, 15)
    epoch_C = 500
    margin_list = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4]
    feat1 = 256
    feat2 = 256
    lr1 = 0.01
    lr2 = 0.001

    for epoch in epoch_list:
        for margin in margin_list:
            print("================================== Parameter =============================")
            print("epoch:{}, margin:{}, feat1:{}, feat2:{}, lr1:{}".format(epoch, margin, feat1, feat2, lr1))
            for i in range(25):
                acc_list = trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, lr2, PLOT=False)
                hist[i] = acc_list
            print("Accuracy: {}, Index: {}".format(hist.mean(axis=0).max(), hist.mean(axis=0).argmax()))
            time.sleep(epoch*5*margin)

