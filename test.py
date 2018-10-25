import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg
import numpy as np
from BPdata import *
from utils import *
dataSet = BPDataSet()
graphSize = dataSet.getGraphSize()
dataSize = dataSet.getDataSize()
data = dataSet.getData()
DTI = data["DTI"]
label = data["label"]
signal = np.random.randn(graphSize)

DTI[DTI<1000] = 0

embeddings = np.zeros((dataSize, graphSize))

for i in range(dataSize):
    L = laplacian(DTI[i])
    q, U = fourier(L)
    vec = np.matmul(U.T, signal)
    embeddings[i] = vec

t_SNE(embeddings, label.reshape(-1))

