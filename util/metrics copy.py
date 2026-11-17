import math, torch
import numpy as np
from collections import defaultdict

class RecommendMetric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = list(res[user].keys())
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(hr_num, user_num):
        return hr_num/user_num

    @staticmethod
    def precision(hits_num, N, user_num):
        return hits_num / (user_num * N)

    @staticmethod
    def recall(hits, origin):
        r=0
        for user in hits:
            r+=hits[user]/len(origin[user])
        return r/len(origin)

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user][:N]):
                if item in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            # for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

def ranking_evaluation(origin, res, N):
    measure = []
    hits=defaultdict(dict)
    real_num = 0 # 记录每个top-n中用户实际交互的物品数量的总和
    for user in res:
        # predicted[user] = res[user][:n]
        real_num += len(origin[user])
        for n in N:
            hits[n][user] = len(set(res[user][:n]) & set(origin[user].keys()))

    for n in N:
        # predicted = {}
        indicators = []
        if len(origin) != len(res):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        # hits = RecommendMetric.hits(origin, resa
        temp=list(hits[n].values())
        hits_num = sum(temp)
        hr_num = np.count_nonzero(np.array(hits.values()))
        hr = RecommendMetric.hit_ratio(hr_num, len(origin))
        indicators.append(f'HR@{n}:' + str(round(hr, 10)))
        prec = RecommendMetric.precision(hits_num, n, len(origin))
        indicators.append(f'Prec@{n}:' + str(round(prec,10)))
        recall = RecommendMetric.recall(hits[n], origin) # 命中总数/真实交互总数
        indicators.append(f'Recall@{n}:' + str(round(recall,10)))
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')

        NDCG = RecommendMetric.NDCG(origin, res, n)
        indicators.append(f'NDCG@{n}:' + str(round(NDCG,10)))
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append(indicators)
    return measure

def rating_evaluation(res):
    measure = []
    mae = RecommendMetric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = RecommendMetric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure


class AttackMetric(object):
    """
    param 
    targetItem:list, targetItem: id
    """
    def __init__(self, recommendModel, targetItem, top=[10]):
        self.recommendModel = recommendModel
        self.targetItem = targetItem
        self.top = top

    def precision(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += k
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def hitRate(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        with torch.no_grad():
            interproduct = (self.recommendModel.user_emb @ self.recommendModel.item_emb.T).cpu().numpy()
        for i in self.recommendModel.data.user:
            if not i.startswith("fakeuser"):
                continue
            u = self.recommendModel.data.user[i]
            score = interproduct[u]
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += 1
                hit[n] += int(len(set(self.targetItem) & set(result[n])) > 0)/len(self.targetItem)

            # for k in range(len(self.top)):
            #     hit[k] += int(len(set(self.targetItem) & set(result[k])) > 0)/len(self.targetItem)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i])
        return result

    def recall(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += len(self.targetItem)
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def NDCG(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                idcg=0
                for s in range(k):
                    if s < len(self.targetItem):
                        idcg+= 1 / np.log2(2 + s)
                totalNum[n] += idcg
            for step,r in enumerate(result):
                for rank,j in enumerate(r):
                    if j in self.targetItem:
                        hit[step]+=1 / np.log2(2 + rank)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result
