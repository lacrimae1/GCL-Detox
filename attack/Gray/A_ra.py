import numpy as np
import random
import torch
from util.tool import targetItemSelect
from util.metrics import AttackMetric
import scipy.sparse as sp
from copy import deepcopy

class A_ra():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.attackModelName = "A_ra"
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
        self.maliciousUserSize = [int(num) / 100 for num in arg.maliciousUserSize.split(',')]
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize[0] < 1:
            self.fakeUserNums = [int(data.user_num * self.maliciousUserSize[i]) for i in
                                 range(len(self.maliciousUserSize))]
        else:
            self.fakeUserNums = int(self.maliciousUserSize)

        self.sigma = 1
        self.n = 100
        self.batchSize = 128

    def posionDataAttack(self, recommender, attack=None):
        self.fakeUserInject(recommender, attack)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
        for epoch in range(self.Epoch):
            # outer optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNums[attack] + self.itemNum, self.userNum + self.fakeUserNums[attack] + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNums[attack], self.userNum + self.fakeUserNums[attack]:] = uiAdj2
            tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                # Pu, Pi = tmpRecommender.model()
                # We do not know Pu, so learn Pu
                optimizer = torch.optim.Adam([tmpRecommender.model.embedding_dict["user_emb"]], lr=recommender.args.lRate)
                tmpRecommender.train(Epoch=5, optimizer=optimizer, evalNum=5)

                _, Pi = tmpRecommender.model()
                approximateUserEmb = (torch.randn((self.n, Pi.shape[1])) * self.sigma).cuda()
                loss = 0
                for i in self.targetItem:
                    for j in range(self.n):
                        loss += -torch.log(torch.sigmoid(approximateUserEmb[j, :] @ Pi[i, :].T) + 10e-8)
                loss = loss.mean()

                optimizer_attack.zero_grad()
                loss.backward()
                optimizer_attack.step()
            Pu, Pi = tmpRecommender.model()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            uiAdj2[self.fakeUser, :] = self.project(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem] = 1

            uiAdj = uiAdj2[:, :]

            # inner optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNums[attack] + self.itemNum, self.userNum + self.fakeUserNums[attack] + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNums[attack], self.userNum + self.fakeUserNums[attack]:] = uiAdj

            recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            recommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=1)

            measure, _, _, hrs = recommender.test(recommender.data.test_set, flag=True)
            # for t, i in enumerate(measure):
            #     logger.info(f"Top{i}:{measure[i]} " + f"targetHR:{hrs[t]}")
            targetHitRate = hrs[0]
            if targetHitRate > bestTargetHitRate:
                bestAdj = uiAdj[:, :]
                bestTargetHitRate = targetHitRate
            uiAdj = bestAdj[:, :]
            
            print("BiLevel epoch {} is over\n".format(epoch + 1))
        self.interact = bestAdj
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

    def fakeUserInject(self, recommender, attack=None):
        Pu, Pi = recommender.model.embedding_dict['user_emb'], recommender.model.embedding_dict['item_emb']
        recommender.data.user_num += self.fakeUserNums[attack]
        for i in range(self.fakeUserNums[attack]):
            recommender.data.user["fakeuser_{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser_{}".format(i)

        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNums[attack]))
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem = random.sample(set(list(range(self.itemNum))), self.maliciousFeedbackNum)
            for i in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u], recommender.data.id2item[i]))
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]

        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)

        recommender.__init__(recommender.args, recommender.data, self.attackModelName)
        with torch.no_grad():
            recommender.model.embedding_dict['user_emb'][:self.userNum, :] = Pu
            recommender.model.embedding_dict['item_emb'][:] = Pi

        recommender.model = recommender.model.cuda()