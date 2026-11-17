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

class SimGCL():
    def __init__(self, args, data, attackName, method=[], T1 = None,T2=None):
        print("Recommender: SimGCL")
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
        if T1 is not None: 
            self.dataName_type += '_' + str(self.T1)
        else:
            self.dataName_type += '_origin'

        # Hyperparameter
        # SimGCL=-n_layer 2 -lambda 0.5 -eps 0.1
        self.max_stop = 20
        self.n_layers = 2
        self.cl = 0.2
        self.bpr = 1
        self.stage = 0
        self.joint = True
        self.eps = 0.1
        self.model = SimGCL_Encoder(self.data, self.args.emb_size, self.eps, self.n_layers)
        self.flag=False
        self.robust = RobustLossComputer(self.model, InfoNCE, bpr_loss, l2_reg_loss, reg=self.args.reg)
        self.T1 = T1
        self.T2 = T2

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
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        else:
            self.optimizer = optimizer
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
        # fake_id = self.get_fake_id()
        for epoch in range(self.maxEpoch):
            if early_stop >= self.max_stop and not is_first:
                print(f"{emoji.emojize(':check_mark_button:')}{'-'*10}Early stopping{'-'*10}{emoji.emojize(':check_mark_button:')}")
                break
            with tqdm(total=len(self.data.training_data), desc=f"Epoch:{epoch+1}/{self.maxEpoch}", ncols=130) as t:
                b = [[] for i in range(2)]
                if "ib" in self.method:
                    self.joint = False
                    if epoch < self.T1:
                        self.bpr = 0
                        self.cl = 0.2
                        self.stage = 1
                        epoch_args = {'stage':self.stage}
                    elif epoch >= self.T1 and epoch<self.T1+self.T2:
                        λ_schedule = np.linspace(0.0, 1.0, self.T2)
                        self.stage = 2
                        self.robust.set_anchor()
                        epoch_args = {'stage':self.stage,'lambda':λ_schedule[(epoch-self.T1)%self.T2],'alpha':0.01}
                    else:
                        self.stage = 3
                        self.bpr = 1
                        self.cl = 0
                        epoch_args = {'stage':self.stage,'gamma':0.3,'alpha':0.01}
                for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    model.train()
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
                    if not self.joint:
                        batch_loss, rec_loss, cl_loss = self.robust.compute(batch, epoch_args)
                    else:
                        cl_loss = self.cl * model.cal_cl_loss([user_idx, pos_idx])
                        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                        batch_loss = rec_loss + l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                    b[0].append(batch_loss.item())
                    b[1].append(rec_loss.item())
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    batch_loss.backward()

                    if requires_adjgrad and self.maxEpoch - epoch < gradIterationNum:
                        self.Matgrad += self.model.sparse_norm_adj.grad
                    elif requires_embgrad and self.maxEpoch - epoch < gradIterationNum:
                        self.usergrad += self.model.embedding_dict["user_emb"].grad
                        self.itemgrad += self.model.embedding_dict["item_emb"].grad

                    self.optimizer.step()
                    t.update(self.args.batch_size)
                    t.set_postfix(batch_loss=batch_loss.item(), rec_loss=rec_loss.item(),
                                   cl_loss=cl_loss.item(),best_epoch=self.best_epoch,stage=f"{self.T1}/{self.T2}/{self.stage}")
                if picture is not None:
                    self.flag=True
                    picture.update(np.mean(b,axis=1),label=["batch","rec"])

                model.eval()
                with torch.no_grad():
                    self.user_emb, self.item_emb = model()
                if epoch % evalNum == 0:
                    if 'ib' in self.method:
                        measure, flag = self.evaluate(epoch)
                        if epoch>self.T1+self.T2:
                            is_first = False
                            if flag:
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

    def test(self, set_type, flag=False,is_pop=False):
        if is_pop: self.data.category()
        # predict
        rec_list = {}
        stds_all = []
        stds_k = defaultdict(list)
        temp = ["popular","unpopular","normal"]
        self.frequency = defaultdict(int)
        total_fre = np.array([0,0,0])
        target_HR = np.zeros(len(self.topN))
        for i, user in enumerate(set_type):
            frequency = np.array([0,0,0])
            candidates = self.predict(user)
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            
            scores, ids = torch.topk(candidates, self.max_N)
            # ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids.numpy()]
            if is_pop:
                for iid in ids.numpy():
                    if self.data.id2item[iid] in set_type[user].keys():
                        if iid in self.data.pop:
                            frequency[0]+=1
                        elif iid in self.data.unpop:
                            frequency[1]+=1
                        else:
                            frequency[2]+=1
                frequency = frequency / (len(set_type[user]))
                total_fre = total_fre + frequency
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
        if is_pop:
            for t, v in enumerate(temp):
                self.frequency[v] = total_fre[t] / len(set_type)
        target_HR = target_HR / np.array([len(set_type)*len(self.targetItem)]*len(self.topN))
        if flag:
            return ranking_evaluation(set_type, rec_list, self.topN), stds_all, stds_k, target_HR
        else:
            return ranking_evaluation(set_type, rec_list, self.topN)


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

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

    def attack_emb(self, users_emb_grad, items_emb_grad):
        with torch.no_grad():
            self.embedding_dict['user_emb'] += users_emb_grad
            self.embedding_dict['item_emb'] += items_emb_grad

    def similarity_all(self, ratio):
        with torch.no_grad():
            # self.thre = torch.cos(torch.tensor(3*torch.pi/180))
            self.k = 1 + int(ratio*5)
            user_embedding, item_embedding = self.forward()
            user_embedding, item_embedding = F.normalize(user_embedding,dim=1), F.normalize(item_embedding,dim=1)
            user_user = user_embedding @ user_embedding.T
            _, user_index = torch.topk(user_user, k=self.k, dim=1)
            index = [[],[]]
            for i, t in enumerate(user_index):
                index[0].extend([i]*self.k)
                index[1].extend(t)
            self.uu = torch.sparse_coo_tensor(torch.tensor(index), torch.tensor([1]*len(index[0])),size=[user_user.shape[0],user_user.shape[0]]).cuda().to_dense()
            
            item_item = item_embedding @ item_embedding.T
            _, item_index = torch.topk(item_item, k=self.k, dim=1)
            index = [[],[]]
            for i, t in enumerate(item_index):
                index[0].extend([i]*self.k)
                index[1].extend(t)
            self.ii = torch.sparse_coo_tensor(torch.tensor(index), torch.tensor([1]*len(index[0])),size=[item_item.shape[0],item_item.shape[0]]).cuda().to_dense()
            
            # uu = ((user_embedding @ user_embedding.T)>self.thre).int().cuda().to(dtype=torch.float32)
            # ii = ((item_embedding @ item_embedding.T)>self.thre).int().cuda().to(dtype=torch.float32)
            self.uu=self.uu
            self.ii=self.ii
            return len(torch.where(self.uu==1)[0]),len(torch.where(self.ii==1)[0])

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        uu, ii = None, None
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2, ee=uu)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2, ee=ii)
        return user_cl_loss + item_cl_loss
    
    def laplacian(self, item_index, embedding):
        W = self.data.laplacian_W(item_index)
        diag = torch.sum(W,dim=1)
        D = torch.diag(diag)
        D_ = torch.diag(torch.pow(diag,-1/2))
        L = D - W
        L = D_ @ L @ D_
        target1 = (embedding.T @ L @ embedding).trace()
        return target1


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
