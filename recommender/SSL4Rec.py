import torch, emoji
import torch.nn as nn
import torch, emoji
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss, InfoNCE
from util.metrics import ranking_evaluation
from tqdm import tqdm
from collections import defaultdict
from util.tool import visualize, Picture
from util.Three_stage import RobustLossComputer


class SSL4Rec():
    def __init__(self, args, data, attackName, method=[], W = None):
        print("Recommender: SSL4Rec")
        self.data = data
        self.args = args
        self.bestPerformance = []
        self.recOutput = []
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.dataName = args.dataset
        self.best_epoch = 0
        self.method = method
        self.ModelName = "SimGCL"
        self.attackName = attackName
        self.dataName_type = "_" + "_".join(method) if len(method)!=0 else ""
        if W is not None: 
            self.dataName_type += '_' + str(self.W)
        else:
            self.dataName_type += '_origin'

        # Hyperparameter
        # SimGCL=-n_layer 2 -lambda 0.5 -eps 0.1
        self.max_stop = 20 if len(self.method)==0 else 30
        self.n_layers = 2
        self.cl = 0.2
        self.bpr=1
        self.stage=0
        self.joint = True
        self.tau = 0.2
        self.drop_rate = 0.2
        self.model = DNN_Encoder(self.data, self.args.emb_size, self.drop_rate, self.tau, self.n_layers)
        self.flag=False
        self.W = W
        self.T2 = 20

    def add(self, targetItem, target):
        self.targetItem = targetItem
        self.target = target
    
    def get_fake_id(self):
        import re
        fake_id = []
        for entry in self.data.training_data:
            user, item, rating = entry
            if re.match(r"fake",user):
                fake_id.append(self.data.user[user])
        return fake_id

    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None,
               evalNum=5,picture=None, attack = 0):
        self.bestPerformance=[]
        model = self.model.cuda()
        if optimizer is None: 
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        else:
            optimizer = optimizer
        if requires_embgrad:
            model.requires_grad = True
            self.usergrad = torch.zeros((self.data.user_num, self.args.emb_size)).cuda()
            self.itemgrad = torch.zeros((self.data.item_num, self.args.emb_size)).cuda()
        elif requires_adjgrad:
            self.model.sparse_norm_adj.requires_grad = True
            self.Matgrad = torch.zeros(
                (self.data.user_num + self.data.item_num, self.data.user_num + self.data.item_num)).cuda()
        self.maxEpoch = self.args.maxEpoch
        if Epoch: self.maxEpoch = Epoch
        early_stop = 0
        is_first = True
        re_best = False
        for epoch in range(self.maxEpoch):
            if early_stop >= self.max_stop and not is_first:
                print(f"{emoji.emojize(':check_mark_button:')}{'-'*10}Early stopping{'-'*10}{emoji.emojize(':check_mark_button:')}")
                break
            with tqdm(total=len(self.data.training_data), desc=f"Epoch:{epoch+1}/{self.maxEpoch}", ncols=130) as t:
                b = [[] for i in range(2)]
                
                if "ib" in self.method:
                    self.joint = False
                    if epoch < self.W:
                        self.bpr = 0
                        self.cl = 0.2
                        self.stage = 1
                        epoch_args = {'stage':self.stage}
                    elif epoch >= self.W and epoch<self.W+self.T2:
                        λ_schedule = np.linspace(0.0, 1.0, self.T2)
                        self.stage = 2
                        self.robust.set_anchor()
                        epoch_args = {'stage':self.stage,'lambda':λ_schedule[(epoch-self.W)%self.T2],'alpha':0.01}
                    else:
                        self.stage = 3
                        if not re_best:
                            self.bestPerformance = []
                            re_best = True
                        self.bpr = 1
                        self.cl = 0
                        epoch_args = {'stage':self.stage,'gamma':0.3,'alpha':0.01}
                for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    model.train()
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    
                    if not self.joint:
                        batch_loss, rec_loss, cl_loss = self.robust.compute(batch, epoch_args)
                    else:
                        cl_loss = self.cl * model.cal_cl_loss([user_idx, pos_idx])
                        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                        batch_loss = rec_loss + l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                    b[0].append(batch_loss.item())
                    b[1].append(rec_loss.item())
                    # Backward and optimize
                    optimizer.zero_grad()
                    batch_loss.backward()

                    if requires_adjgrad and self.maxEpoch - epoch < gradIterationNum:
                        self.Matgrad += self.model.sparse_norm_adj.grad
                    elif requires_embgrad and self.maxEpoch - epoch < gradIterationNum:
                        self.usergrad += self.model.embedding_dict["user_emb"].grad
                        self.itemgrad += self.model.embedding_dict["item_emb"].grad

                    optimizer.step()
                    t.update(self.args.batch_size)
                    t.set_postfix(batch_loss=batch_loss.item(), rec_loss=rec_loss.item(),
                                  cl_loss=cl_loss.item(), best_epoch=self.best_epoch,stage=f"{self.W}/{self.stage}")
                if picture is not None:
                    self.flag=True
                    picture.update(np.mean(b,axis=1),label=["batch","rec"])
                model.eval()
                with torch.no_grad():
                    self.user_emb, self.item_emb = self.model()
                if epoch % evalNum == 0:
                    measure, flag = self.evaluate(epoch)
                    best = ':love_letter:' if flag else ""
                    t.write(f"{emoji.emojize(best)}Epoch:{epoch+1}/{self.maxEpoch}.Evaluating The Model...\
                            {emoji.emojize(':face_with_monocle:')}\nRecommender Model:{self.args.model_name}")
                    for i in measure:
                        t.write(f"Top{i}:{measure[i]}")
                if epoch % evalNum == 0:
                    if 'ib' in self.method:
                        measure, flag = self.evaluate(epoch)
                        if self.stage==3:
                            if flag:
                                if epoch > self.W +self.T2+ 1:
                                    is_first = False
                                early_stop = 0
                            else:
                                early_stop += 1
                        best = ':love_letter:' if flag else ""
                        t.write(f"{emoji.emojize(best)}Epoch:{epoch + 1}/{self.maxEpoch}.Evaluating The Model...\
                                {emoji.emojize(':face_with_monocle:')}\nRecommender Model:{self.args.model_name}")
                        for i in measure:
                            t.write(f"Top{i}:{measure[i]}")
                    else:
                        measure, flag = self.evaluate(epoch)
                        if flag:
                            if epoch > 0:
                                is_first = False
                            early_stop = 0
                        else:
                            early_stop += 1
                        best = ':love_letter:' if flag else ""
                        t.write(f"{emoji.emojize(best)}Epoch:{epoch + 1}/{self.maxEpoch}.Evaluating The Model...\
                                {emoji.emojize(':face_with_monocle:')}\nRecommender Model:{self.args.model_name}")
                        for i in measure:
                            t.write(f"Top{i}:{measure[i]}")
                    
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        # self.measure, self.stds, self.targetHR = self.test(self.data.test_set, flag = True)
        self.measure, self.stds_all, self.stds_k, self.targetHR = self.test(self.data.test_set, flag = True)
        saved = {"measure":self.measure, "stds_all":self.stds_all, "stds_k":self.stds_k, "targetHR":self.targetHR, "best_epoch":self.best_epoch,
                "user_emb":self.user_emb, "item_emb":self.item_emb, "emb_dict":self.best_emb_dict}
        print(f"{emoji.emojize(':hundred_points:')} Test Performance {emoji.emojize(':hundred_points:')}")
        for t,i in enumerate(self.measure):
            print(f"Top{i}:{self.measure[i]} "+f"targetHR:{self.targetHR[t]}")
        return saved

    def save(self, epoch):
        with torch.no_grad():
            self.best_emb_dict = self.model.embedding_dict
            self.best_user_emb, self.best_item_emb = self.model.forward()
            self.best_model = self
            self.best_epoch = epoch+1

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

    def test(self, set_type, flag=False,attack=None):
        if attack is None: self.data.category()
        # predict
        rec_list = {}
        stds_all = []
        stds_k = defaultdict(list)
        self.frequency = defaultdict(int)
        target_HR = np.zeros(len(self.topN))
        for i, user in enumerate(set_type):
            candidates = self.predict(user)
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            
            scores, ids = torch.topk(candidates, self.max_N)
            # ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids.numpy()]
            if attack is None:
                for iid in ids.numpy():
                    if iid in self.data.pop:
                        self.frequency["popular"]+=1
                    elif iid in self.data.unpop:
                        self.frequency["unpopular"]+=1
                    else:
                        self.frequency["normal"]+=1
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

class DNN_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, temperature, n_layers):
        super(DNN_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.tau = temperature
        self.user_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(drop_rate)
        self._init_model()

        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        
        # initializer = nn.init.xavier_uniform_

        # self.initial_user_emb = nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size)))
        # self.initial_item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size)))

    def _init_uiAdj(self, ui_adj):
        D = sp.diags(np.array((1 / np.sqrt(ui_adj.sum(1)))).flatten())
        self.sparse_norm_adj = D @ ui_adj @ D
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.sparse_norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
    
    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        self.user_all_embeddings, self.item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return self.user_all_embeddings, self.item_all_embeddings

    def item_encoding(self,x1, x2):
        u_emb = self.user_all_embeddings[x1]
        u1_emb = self.dropout(u_emb)
        u2_emb = self.dropout(u_emb)

        i_emb = self.item_all_embeddings[x2]
        i1_emb = self.dropout(i_emb)
        i2_emb = self.dropout(i_emb)
        return u1_emb, u2_emb, i1_emb, i2_emb

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view1, user_view_2, item_view1, item_view_2 = self.item_encoding(u_idx, i_idx)
        cl_loss = InfoNCE(item_view1, item_view_2, self.tau) + InfoNCE(user_view1, user_view_2, self.tau)
        return cl_loss

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
