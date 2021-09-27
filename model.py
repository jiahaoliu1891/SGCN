# coding=utf-8

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from utils import *
from config import *

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.xavier_normal_(torch.empty(in_features, out_features)))
        if bias:
            self.bias = Parameter(init.constant_(torch.empty(out_features), 0.1))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        # output = support
        if self.bias is not None:
            out = output + self.bias
            return out
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def Korder(data, k):
    if k == 0:
        r = torch.Tensor(np.zeros_like(data))
        a = np.arange(r.size(-1))
        r[:, a, a] = 1.0
    else:
        r = data
        for i in range(k-1):
            r = r @ data
    return r


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.classify = nn.Linear(nfeat, 1)

    def forward(self, x, adj):
        batch_size = adj.size(0)
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        out = x.view(batch_size, -1)
        return out

    def forwardSave(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

    def forwardDownSample(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x



class SiameseGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(SiameseGCN, self).__init__()
        self.gcn = GCN(nfeat, nhid, nclass)

    def forward_once(self, x, adj):
        return self.gcn(x, adj)

    def forwardSave(self, x, adj):
        out = self.gcn.forwardSave(x, adj)
        for i in range(out.shape[0]):
            np.savetxt("./embedding/embedding{}.csv".format(i), out[i].detach().numpy(), delimiter=',')

    def forward(self, x1, adj1, x2, adj2):
        out1 = self.forward_once(x1, adj1)
        out2 = self.forward_once(x2, adj2)
        return out1, out2

    def downSample(self):
        pass

    def forwardDownSample(self, x, adj):
        batchSize = x.shape[0]
        graphSize = x.shape[-1]
        out = self.gcn.forwardDownSample(x, adj)
        adj_sum = torch.sum(adj, dim=0)
        degree = compute_degree(adj_sum)
        idx = nsmall_index(degree, NSMALL)
        whole_idx = np.arange(graphSize)
        return_idx = np.setdiff1d(whole_idx, idx)
        out = out[:, return_idx, :]
        out = out.view(batchSize, -1)
        return out


class AddSiameseGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(AddSiameseGCN, self).__init__()
        self.gcn = GCN(nfeat, nhid, nclass)
        self.classify = torch.nn.Linear(nclass*82, 1)

    def forward_once(self, x, adj):
        return self.gcn(x, adj)

    def forwardSave(self, x, adj):
        out = self.gcn.forwardSave(x, adj)
        for i in range(out.shape[0]):
            np.savetxt("./embedding/embedding{}.csv".format(i), out[i].detach().numpy(), delimiter=',')

    def forward(self, x1, adj1, x2, adj2):
        out1 = self.forward_once(x1, adj1)
        out2 = self.forward_once(x2, adj2)
        outClass = torch.sigmoid(self.classify(out1))
        return out1, out2, outClass             

    def test(self, x, adj):
        return torch.sigmoid(self.classify(self.forward_once(x, adj)))


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=4):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, 2)
        label = label.view( label.size()[0] )

        loss_same = label * torch.pow(euclidean_distance, 2)
        loss_diff = (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean(loss_same + loss_diff)

        return loss_contrastive


class AddContrastiveLoss(torch.nn.Module):
    def __init__(self, lamb=1, margin=1.5):
        super(AddContrastiveLoss, self).__init__()
        self.margin = margin
        self.lamb = lamb
        self.loss_function = nn.BCELoss()

    def forward(self, output1, output2, output_class, siamese_label, class_label):
        euclidean_distance = F.pairwise_distance(output1, output2, 2)
        label = siamese_label.view( siamese_label.size()[0] )

        loss_same = label * torch.pow(euclidean_distance, 2)
        loss_diff = (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # print("same{}\t\tdiff{}".format(torch.mean(loss_same), torch.mean(loss_diff)))

        loss_contrastive = torch.mean(loss_same + loss_diff)
        loss_classify = self.loss_function(output_class, class_label)

        print("loss_classify{}\t\tloss_contrastive{}".format(loss_classify, loss_contrastive))

        loss = loss_contrastive + loss_classify
        return loss


class ClassifyNetwork(nn.Module):
    def __init__(self, nfeat):
        super(ClassifyNetwork, self).__init__()
        self.classify = torch.nn.Linear(nfeat, 1)

    def forward(self, W):
        return torch.sigmoid(self.classify(W))


if __name__ == "__main__":
    a = torch.Tensor([[1, 2], [3, 4]])

