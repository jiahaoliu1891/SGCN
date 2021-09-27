# coding=utf-8

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import heapq
import torch


def laplacian(W, normalize=True):
    N = W.shape[0]
    assert N == config.graphsize
    d = np.sum(W, axis=0)
    d[d == 0] = 0.1
    if not normalize:
        D = np.diag(d)
        L = D - W
    else:
        d_sqrt = np.sqrt(d)
        d = 1 / d_sqrt
        D = np.diag(d)
        I = np.diag(np.ones(N))
        L = I - np.dot(np.dot(D, W), D)
    return L


def fourier(L):
    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]
    lamb, U = np.linalg.eig(L)
    lamb, U = sort(lamb, U)
    return lamb, U


def t_SNE(X, y):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, perplexity=40)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    plt.figure(figsize=(6, 6))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], s=150, c=y, alpha=.7)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def compute_degree(data):
    data = data.detach() 
    degree = data.cpu().numpy().copy() if torch.cuda.is_available() else data.numpy().copy()
    degree = np.sum(degree, axis=1)
    return degree


def nlarge_index(data, n):
    return heapq.nlargest(n, range(len(data)), data.take)


def nsmall_index(data, n):
    return heapq.nsmallest(n, range(len(data)), data.take)
