import torch, emoji, time
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss
from util.metrics import ranking_evaluation
from tqdm import tqdm
from collections import defaultdict

class LightGCN():
    def __init__(self, args, data, attackModelName, method=[]):
        print("Recommender: LightGCN")
        self.data = data
        self.args = args
        self.bestPerformance = []
        self.recOutput = []
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model = LGCN_Encoder(self.data, self.args.emb_size, self.args.n_layers)
        self.dataName = args.dataset
        self.best_epoch = 0
        self.flag=False
        self.max_stop = 20
        self.attackModelName = attackModelName

    def add(self, targetItem, target):
        self.targetItem = targetItem
        self.target = target

    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None,
               evalNum=10, picture=None, attack=0):
        self.bestPerformance=[]
        model = self.model.cuda()
        if optimizer is None: 
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        else:
            self.optimizer = optimizer
        early_stop = 0
        is_first = True
        self.maxEpoch = self.args.maxEpoch
        if Epoch: self.maxEpoch = Epoch
        for epoch in range(self.maxEpoch):
            if early_stop >= self.max_stop and not is_first:
                print(f"{emoji.emojize(':check_mark_button:')}{'-' * 10}Early stopping{'-' * 10}{emoji.emojize(':check_mark_button:')}")
                break
            with tqdm(total=len(self.data.training_data), desc=f"Epoch:{epoch+1}/{self.maxEpoch}", ncols=50) as t:
                b = [[]]
                for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    model.train()
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
                    batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.args.reg, user_emb,
                                                                                            pos_item_emb)
                    b[0].append(batch_loss.item())
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    t.update(self.args.batch_size)
                    t.set_postfix(loss=batch_loss.item(),best_epoch=self.best_epoch)
                if picture is not None:
                    self.flag=True
                    picture.update(np.mean(b, axis=1))
                model.eval()
                with torch.no_grad():
                    self.user_emb, self.item_emb = model()
                if epoch % evalNum == 0:
                    measure, flag = self.evaluate(epoch)
                    if flag:
                        early_stop = 0
                    else:
                        early_stop += 1
                    best = ':love_letter:' if flag else ""
                    t.write(f"{emoji.emojize(best)}Epoch:{epoch+1}/{self.maxEpoch}.Evaluating The Model...\
                            {emoji.emojize(':face_with_monocle:')}\nRecommender Model:{self.args.model_name}")
                    for i in measure:
                        t.write(f"Top{i}:{measure[i]}")
                # self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        self.measure, self.stds_all, self.stds_k, self.targetHR = self.test(self.data.test_set, flag = True)
        saved = {"measure":self.measure, "stds_all":self.stds_all, "stds_k":self.stds_k, "targetHR":self.targetHR,
                 "best_epoch":self.best_epoch,
                "user_emb":self.user_emb, "item_emb":self.item_emb, "emb_dict":self.best_model.model.embedding_dict}
        print(f"{emoji.emojize(':hundred_points:')} Test Performance {emoji.emojize(':hundred_points:')}")
        for i in self.measure:
            print(f"Top{i}:{self.measure[i]}")
        return saved

    def save(self, epoch):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            self.best_model = self
            if self.flag: self.best_epoch = epoch+1

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu()

    def evaluate(self, epoch):
        # print('Evaluating the model...')
        measure_all = self.test(self.data.val_set)
        flag = False
        if len(self.bestPerformance) > 0:
            count = 0
            performance = measure_all[self.max_N]
            for k in list(self.bestPerformance[1].keys())[:2]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.best = measure_all
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save(epoch)
                flag = True
        else:
            self.best = measure_all
            self.bestPerformance.append(epoch + 1)
            performance = measure_all[self.max_N]
            self.bestPerformance.append(performance)
            self.save(epoch)
            flag = True
        return measure_all, flag

    def test(self, set_type, flag=False):
        # predict
        rec_list = {}
        stds_all = []
        stds_k = defaultdict(list)
        target_HR = np.zeros(len(self.topN))
        for i, user in enumerate(set_type):
            candidates = self.predict(user)
            rated = self.data.user_rated_id(user)
            candidates[rated] = -10e8
            scores, ids = torch.topk(candidates, self.max_N)
            # ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids.numpy()]
            if flag:
                for i, n in enumerate(self.topN):
                    target_HR[i] += len(set(item_names[:n]) & set(self.targetItem))
            rec_list[user] = list(zip(item_names, scores))
            # rec_list[user] = ids.numpy()
            if flag:
                for n in self.topN:
                    stds_k[n].append(torch.std(scores[:n]).numpy())
                std = torch.std(scores).numpy()
                stds_all.append(std)
        target_HR = target_HR / np.array([len(set_type)*len(self.targetItem)]*len(self.topN))
        if flag:
            return ranking_evaluation(set_type, rec_list, self.topN), stds_all, stds_k, target_HR
        else:
            return ranking_evaluation(set_type, rec_list, self.topN)


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_uiAdj(self, ui_adj):
        D = sp.diags(np.array((1 / np.sqrt(ui_adj.sum(1)))).flatten())
        self.sparse_norm_adj = D @ ui_adj @ D
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.sparse_norm_adj).cuda()

    def attack_emb(self, users_emb_grad, items_emb_grad):
        with torch.no_grad():
            self.embedding_dict['user_emb'] += users_emb_grad
            self.embedding_dict['item_emb'] += items_emb_grad

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
