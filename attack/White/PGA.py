import numpy as np
import random, time
import torch
import torch.nn as nn
from util.tool import targetItemSelect
from util.algorithm import find_k_largest
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import scipy.sparse as sp
from copy import deepcopy
from util.loss import bpr_loss, l2_reg_loss
from sklearn.neighbors import LocalOutlierFactor as LOF
from recommender.GMF import GMF
from tqdm import tqdm
import logging


class PGA():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.attackModelName = "PGA"
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]
        self.targetItem_name = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem_name]
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # limitation 
        self.maliciousUserSize = [int(num)/100 for num in arg.maliciousUserSize.split(',')]
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize[0] < 1:
            self.fakeUserNums = [int(data.user_num * self.maliciousUserSize[i]) for i in range(len(self.maliciousUserSize))]
        else:
            self.fakeUserNums = int(self.maliciousUserSize)

        self.batchSize = 256

    def posionDataAttack(self, recommender, attack=None):
        # print(f"recommender所占显存:{recommender.numel()*recommender.element_size() / 1024**3} GB")
        _, maxRecNumItemInd = torch.topk(torch.tensor(self.interact.sum(0)), int(self.itemNum * 0.05))
        self.maxRecNumItemInd = maxRecNumItemInd
        Pu, Pi = recommender.model.embedding_dict['user_emb'], recommender.model.embedding_dict['item_emb']
        optimizer = torch.optim.SGD(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        self.dataUpdate(recommender,attack)
        recommender.__init__(recommender.args, recommender.data, self.attackModelName)
        newAdj = recommender.data.matrix()
        with torch.no_grad():
            recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
            recommender.model.embedding_dict['item_emb'][:] = Pi
        self.controlledUser = list(range(self.userNum, self.userNum + self.fakeUserNums[attack]))
        # recommender.train(Epoch=self.Epoch, optimizer=optimizer, evalNum=2)
        # originRecommender = deepcopy(recommender)
        uiAdj = newAdj[:, :]
        for u in self.controlledUser:
            uiAdj[u, :] = 0
            uiAdj[u, self.targetItem] = 1
            uiAdj[u, maxRecNumItemInd] = torch.rand([1]).item()

        # recommender = deepcopy(originRecommender)
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        for epoch in range(self.outerEpoch):
            # outer optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNums[attack] + self.itemNum, self.userNum + self.fakeUserNums[attack] + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNums[attack], self.userNum + self.fakeUserNums[attack]:] = uiAdj
            recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            recommender.train(Epoch=self.Epoch, optimizer=optimizer, evalNum=3)

            # inner optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            for inner in range(self.innerEpoch):
                users, pos_items, neg_items = [], [], []
                with tqdm(total=self.itemNum, desc=f"innerEpoch:{inner+1}/{self.innerEpoch}",ncols=120) as t:
                    for batch in range(0,self.itemNum,self.batchSize):
                        start=time.time()
                    ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                        self.userNum + self.fakeUserNums[attack] + self.itemNum, self.userNum + self.fakeUserNums[attack] + self.itemNum),
                                        dtype=np.float32)
                    ui_adj[:self.userNum + self.fakeUserNums[attack], self.userNum + self.fakeUserNums[attack]:] = uiAdj2
                    tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
                    tmpRecommender.model.sparse_norm_adj.requires_grad = True

                    Pu, Pi = tmpRecommender.model()
                    if len(users) == 0:
                        scores = torch.matmul(Pu, Pi.transpose(0, 1))
                        _, top_items = torch.topk(scores, 50)
                        top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                        for idx, u_index in enumerate(list(range(self.userNum))):
                            for item in self.targetItem:
                                users.append(u_index)
                                pos_items.append(item)
                                neg_items.append(top_items[u_index].pop())
                    pos_score = torch.mul(Pu[users], Pi[pos_items]).sum(dim=1)
                    neg_score = torch.mul(Pu[users], Pi[neg_items]).sum(dim=1)
                    CWloss = neg_score - pos_score
                    CWloss = CWloss.mean()
                    Loss = CWloss
                    torch.cuda.empty_cache()
                    print(f"torch.autograd.grad前已分配显存:{torch.cuda.memory_allocated()/1024**3} GB")
                    print(f"还剩下{torch.cuda.memory_reserved()/1024**3} GB")
                    doubleGrad = torch.autograd.grad(Loss, tmpRecommender.model.sparse_norm_adj)[0]
                    print(f"torch.autograd.grad前已分配显存:{torch.cuda.memory_allocated()/1024**3} GB")
                    with torch.no_grad():
                        rowsum = np.array((ui_adj+ui_adj.T).sum(1))
                        d_inv = np.power(rowsum, -0.5).flatten()
                        d_inv[np.isinf(d_inv)] = 0.
                        d_mat_inv = sp.diags(d_inv)
                        indices = torch.tensor([list(range(d_mat_inv.shape[0])), list(range(d_mat_inv.shape[0]))])
                        values = torch.tensor(d_inv, dtype=torch.float32)
                        d_mat_inv = torch.sparse_coo_tensor(indices=indices, values=values, size=[d_mat_inv.shape[0], d_mat_inv.shape[0]]).cuda()
                        norm_adj_tmp = torch.sparse.mm(d_mat_inv,doubleGrad)
                        doubleGrad = torch.sparse.mm(norm_adj_tmp,d_mat_inv)
                    doubleGrad = doubleGrad.to_dense()
                    grad = doubleGrad[
                        :self.userNum + self.fakeUserNums[attack],
                        self.userNum + self.fakeUserNums[attack]:][self.controlledUser, :] + doubleGrad[
                                                                                    self.userNum + self.fakeUserNums[attack]:,
                                                                                    :self.userNum + self.fakeUserNums[attack]].T[
                                                                                    self.controlledUser, :]
                    with torch.no_grad():
                        subMatrix = torch.tensor(uiAdj2[self.controlledUser, :].todense()).cuda()
                        subMatrix -= 0.2 * torch.tanh(grad)
                        subMatrix[subMatrix > 1] = 1
                        subMatrix[subMatrix <= 0] = 10e-8
                        uiAdj2[self.controlledUser, :] = subMatrix.cpu()
                    # end = time.time()
                    # res=(end-start)
                    # t.update(self.batchSize)
                    # t.set_postfix(outerEpoch=f"{epoch+1}/{self.outerEpoch}" , run_time_for_batch=f"{res:.4f}s.")
                    print(f"del前已分配显存:{torch.cuda.memory_allocated()/1024**3} GB")
                    del doubleGrad, d_mat_inv, norm_adj_tmp,grad,subMatrix
                    print(f"del后已分配显存:{torch.cuda.memory_allocated()/1024**3} GB")
                    print(f"还剩下{torch.cuda.memory_reserved()/1024**3} GB")
                    print(f"{'-'*10} 第{inner}个innerEpoch完成")
            uiAdj2[self.controlledUser, :] = self.project(uiAdj2[self.controlledUser, :],
                                                          int(self.maliciousFeedbackSize * self.itemNum))
            for u in self.controlledUser:
                for i in self.targetItem:
                    uiAdj2[u, i] = 1
            uiAdj = uiAdj2[:, :]
            print("attack step {} is over\n".format(epoch + 1))
        self.interact = uiAdj
        return self.interact

    def project(self, mat, n):
        try:
            matrix = torch.tensor(mat[:, :].todense())
            _, indices = torch.topk(matrix, n, dim=1)
            matrix.zero_()
            matrix.scatter_(1, indices, 1)
        except:
            matrix = mat[:,:]
            for i in range(matrix.shape[0]):
                subMatrix = torch.tensor(matrix[i, :].todense())
                topk_values, topk_indices = torch.topk(subMatrix, n)
                subMatrix.zero_()  
                subMatrix[0, topk_indices] = 1
                matrix[i, :] = subMatrix[:, :].flatten()
        return matrix

    def dataUpdate(self, recommender, attack=None):
        recommender.data.user_num += self.fakeUserNums[attack]
        for i in range(self.fakeUserNums[attack]):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)
        n_nodes = recommender.data.user_num + recommender.data.item_num
        row_idx = [recommender.data.user[pair[0]] for pair in recommender.data.training_data]
        col_idx = [recommender.data.item[pair[1]] for pair in recommender.data.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + recommender.data.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        recommender.data.ui_adj = adj_mat
        recommender.data.norm_adj = recommender.data.normalize_graph_mat(recommender.data.ui_adj)
        row, col, entries = [], [], []
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]
        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
