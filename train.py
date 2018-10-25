import torch
from torch import optim
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
from BPdata import BPDataSet
from model import *
import scipy.io as sio


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

        print("Epoch number {} ".format(epoch))
        print("Current loss {} Accuracy:{}".format(loss.item(), 0))

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


    # newData = dataSet.getData()
    # newfMRI = newData["fMRI"]
    # newDTI = newData["DTI"]
    # newfMRI, newDTI = map(lambda x: torch.Tensor(x), [newfMRI, newDTI])
    # net.forwardSave(newfMRI, newDTI)

    # np.savetxt("./label.csv", trainLabel.numpy(), delimiter='\n')

    embeddingsTrain = net.forward_once(trainfMRI, trainDTI).detach()
    embeddingsTest = net.forward_once(testfMRI, testDTI).detach()
    classNet = ClassifyNetwork(FEAT2 * graphSize)
    acc = trainClassNet(classNet, embeddingsTrain, embeddingsTest, trainLabel, testLabel, testSize, EPOCH_C, LR2)

    print(acc)
    # if acc > 0.63:
    #     dataSet.makeNewData()
    return acc


def trainClassNet(classNet, embeddingsTrain, embeddingsTest, trainLabel, testLabel, testSize, EPOCH_C, LR2):
    train_size = trainLabel.size(0)
    optm = optim.Adam(classNet.parameters(), lr=LR2)

    def compute_acc():
        X = classNet(embeddingsTest).detach().numpy()
        X[X >= 0.5] = 1
        X[X < 0.5] = 0
        return ((X == testLabel).sum()) / testSize

    loss_function = nn.BCELoss()
    idx = np.arange(train_size)
    acc = 0

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
        print("Epoch number {} ".format(i))
        print("Current loss {} Accuracy:{}".format(Loss.item(), acc))

        if Loss.item() <= 0.1:
            return acc

    return acc


if __name__ == "__main__":

    import time
    #
    # EPOCH = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    # MARGIN = [0.1,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # FEAT1 = [4,8,16,32,64,128,256]
    # FEAT2 = [4,8,16,32,64,128,256]
    # LR1 = [0.002,0.005,0.01]
    # PLOT = False

    # epoch = 10
    # epoch_C = 90
    # margin = 0.5
    # feat1 = 128
    # feat2 = 128
    # lr1 = 0.01

    hist = []
    epoch = 10
    epoch_C = 45
    margin = 1.5
    feat1 = 256
    feat2 = 128
    lr1 = 0.01
    lr2 = 0.001

    print("===========================================参数配置=============================")
    print("epoch:{}, margin:{}, feat1:{}, feat2:{}, lr1:{}".format(epoch, margin, feat1, feat2, lr1))
    # trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, PLOT=False)
    for i in range(25):
        hist.append(trainSiamese(epoch, epoch_C, margin, feat1, feat2, lr1, lr2, PLOT=False))
    print("==================================准确率{}=============================".format(np.mean(hist)))



