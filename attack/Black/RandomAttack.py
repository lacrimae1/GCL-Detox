import numpy as np
import random
from util.tool import targetItemSelect
from scipy.sparse import vstack, csr_matrix


class ShillingAttackModel():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
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
        self.recommenderModelRequired = False

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

    def getPopularItemId(self, N):
        """
        get N popular items based on the number of feedbacks
        :return: id list
        """
        return np.argsort(self.interact[:, :].sum(0))[0, -N:].tolist()[0]

    def getReversePopularItemId(self, N):
        """
        get N unpopular items based on the number of feedbacks
        :return: id list
        """
        return np.argsort(self.interact[:, :].sum(0))[0, :N].tolist()[0]

    def posionDataAttack(self):
        """
        A rating matrix for the fake user segment is generated according to a specific algorithm
        :return: Fake user rating matrix
        """


class RandomAttack(ShillingAttackModel):
    def __init__(self, arg, data):
        super(RandomAttack, self).__init__(arg, data)

    def posionDataAttack(self, attack=None):
        self.fakeUserNum = self.fakeUserNums[attack]
        uNum = self.fakeUserNum
        row, col, entries = [], [], []
        for i in range(uNum):
            # fillerItemid = random.sample(set(range(self.itemNum)) - set(self.targetItem),
            #                              self.maliciousFeedbackNum - len(self.targetItem))
            fillerItemid = random.sample(set(range(self.itemNum)) - set(self.targetItem),
                                self.maliciousFeedbackNum)
            row += [i for r in range(len(fillerItemid + self.targetItem))]
            col += fillerItemid + self.targetItem
            entries += [1 for r in range(len(fillerItemid + self.targetItem))]
        fakeRat = csr_matrix((entries, (row, col)), shape=(uNum, self.itemNum), dtype=np.float32)
        return vstack([self.interact, fakeRat])
