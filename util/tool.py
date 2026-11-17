import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import manifold

class Picture():
    def __init__(self, recommendModelName, datasetName,method, attackModel=None, attackSize="",curveType="Loss_Curve"):
        temp = recommendModelName
        self.method=method
        if attackSize is not None:
            t = "_" + attackSize if len(attackSize)!=0 else ""
            temp += f"_{attackModel}" + t
        self.figure_path = f"./result_show/{attackModel}/{datasetName}/{datasetName+self.method}/{curveType}/{recommendModelName}/"
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)
        self.figure_path += temp
        self.epoch = []
        self.curveType = curveType
        self.num = 0
        self.color = ['red', 'pink', 'blue','orange','cyan']
        self.fig, self.ax = plt.subplots()
    
    def update(self, value, label=["batch"]):
        self.num += 1
        if self.num == 1:
            self.value = [[] for i in range(len(value))]
        self.epoch.append(self.num)
        self.ax.clear()
        for i, l in enumerate(value):
            self.value[i].append(l)
            self.ax.plot(self.epoch, self.value[i], color=self.color[i],label=label[i])
        xlim = [1, self.num + 5]
        self.ax.set_title(self.curveType)
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("value")
        self.ax.legend()
        self.ax.set_xlim(xlim)
        self.fig.savefig(self.figure_path+".png", dpi=300)
        
def visualize(user_emb, item_emb, recommenderName, attackName, dataName_Type, targetItem, attack="", pre="pre_", dataName=""):
    path = f"./result_show/{attackName}/{dataName}/{dataName+dataName_Type}/distribution/"
    if not os.path.exists(path):
        os.makedirs(path)
    with torch.no_grad():
        embedding = torch.cat([user_emb, item_emb],0)
        embedding = embedding.detach().cpu().numpy()
        user = np.array([0]*len(user_emb))
        item = np.array([1]*len(item_emb))
        item[np.array(targetItem)] = 2
        labels = np.concat((user,item))
    tsne = manifold.TSNE(perplexity=100, n_components=2, init='pca',random_state=2025)
    X_tsne = tsne.fit_transform(embedding)
    plt.figure(figsize=(8,8))
    plt.scatter(X_tsne[labels == 0, 0],X_tsne[labels==0, 1],s=10,c='red',label='Users',alpha=0.3)
    plt.scatter(X_tsne[labels == 1, 0],X_tsne[labels==1, 1],s=10,c='blue',label='Items',alpha=0.3)
    plt.scatter(X_tsne[labels == 2, 0],X_tsne[labels==2, 1],s=10,c='black',label='target',alpha=1)
    plt.title(f"{recommenderName}")
    plt.legend()
    plt.savefig(path+pre+f"{recommenderName + attack}.png", dpi=200)

def isClass(obj, classList):
    """
    If obj is in classList, return True
    else return False
    """
    for i in classList:
        if isinstance(obj, i):
            return True
    return False

def getPopularItemId(interact, n):
    """
    Get the id of the top n popular items based on the number of ratings
    :return: id list
    """
    return np.argsort(interact[:, :].sum(0))[-n:]

def dataSave(ratings, fileName, id2user, id2item):
    """
    sava ratings data
    :param ratings: np.array ratings matrix
    :param fileName: str fileName
    :param id2user: dict
    :param id2item: dcit
    """
    ratingList = []
    ind = ratings.nonzero()
    for i,j in zip(ind[0].tolist(),ind[1].tolist()):
            ratingList.append((i,j,ratings[i,j]))
    # for i in range(ratings.shape[0]):
    #     for j in range(ratings.shape[1]):
    #         if ratings[i,j] == 0: continue
    #         ratingList.append((i, j, ratings[i,j]))
    text = []
    for i in ratingList:
        if i[0] in id2user.keys():
            userId = id2user[i[0]]
        else:
            userId = "fakeUser" + str(i[0])
        itemId = id2item[i[1]]
        new_line = '{} {} {}'.format(userId, itemId, i[2]) + '\n'
        text.append(new_line)
    with open(fileName, 'w') as f:
        f.writelines(text)


def targetItemSelect(data, arg, popularThreshold=0.1):
    interact = data.matrix()
    userNum = interact.shape[0]
    itemNum = interact.shape[1]
    targetSize = arg.targetSize
    if targetSize < 1:
        targetNum = int(targetSize * itemNum)
    else:
        targetNum = int(targetSize)
    path = './data/clean/' + data.dataName + "/" + "targetItem_" + arg.attackTargetChooseWay + "_" + str(
        targetNum) + ".txt"
    if os.path.exists(path):
        with open(path, 'r') as f:
            line = f.read()
            targetItem = [i.replace("'", "") for i in line.split(",")]
        return targetItem
    else:
        def getPopularItemId(n):
            """
            Get the id of the top n popular items based on the number of ratings
            :return: id list
            """
            return np.argsort(interact[:, :].sum(0))[0, -n:].tolist()[0]

        def getReversePopularItemId(n):
            """
            Get the ids of the top n unpopular items based on the number of ratings
            :return: id list
            """
            return np.argsort(interact[:, :].sum(0))[0, :n].tolist()[0]

        if arg.attackTargetChooseWay == "random":
            targetItem = random.sample(set(list(range(itemNum))),
                                       targetNum)
        elif arg.attackTargetChooseWay == "popular":
            targetItem = random.sample(set(getPopularItemId(int(popularThreshold * itemNum))),
                                       targetNum)
        elif arg.attackTargetChooseWay == "unpopular":
            targetItem = random.sample(
                set(getReversePopularItemId(int(0.2 * itemNum))),
                targetNum)
            # targetItem = random.sample(
            #     set(getReversePopularItemId(int((1 - popularThreshold) * itemNum))),
            #     targetNum)
        targetItem = [data.id2item[i] for i in targetItem]
        with open(path, 'w') as f:
            f.writelines(str(targetItem).replace('[', '').replace(']', ''))
        return targetItem

def seedSet(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False