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
from util.tool import visualize

class Prefine():
    def __init__(self, args, data, attackName, method=[], W = 100):
        print("Recommender: Prefine")
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
        self.ModelName = "Prefine"
        self.attackName = attackName
        t = "_" + "_".join(method)
        if W != 100: t = t+f"_{W}"
        self.dataName_Type = f"{self.dataName + t}"

        # Hyperparameter
        # SimGCL=-n_layer 2 -lambda 0.5 -eps 0.1
        self.max_stop = 40 if len(self.method)==0 else 50
        self.n_layers = 2
        self.cl = 0.2
        self.bpr = 1
        self.stage = False
        self.joint = True
        self.eps = 0.1
        self.model = SimGCL_Encoder(self.data, self.args.emb_size, self.eps, self.n_layers)
        self.flag=False
        self.W = W

        self.outerEpoch = 1
        self.infoEpoch = 1
        self.bprEpoch = 200

    def add(self, targetItem, target):
        self.targetItem = targetItem
        self.target = target
    
    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None,
               evalNum=5,picture=None, attack = 0):
        self.bestPerformance=[]
        model = self.model.cuda()
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
        for epoch in range(self.outerEpoch):
            optimizer = torch.optim.Adam(model.embedding_dict.parameters(), lr=self.args.lRate)
            for info in range(self.infoEpoch):
                with tqdm(total=len(self.data.training_data), desc=f"infoEpoch:{info+1}/{self.infoEpoch}", ncols=130) as t:
                    b = [[]]
                    for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                        user_idx, pos_idx, neg_idx = batch
                        model.train()
                        rec_user_emb, rec_item_emb = model.info_forward()
                        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                            neg_idx]
                        
                        cl_loss = self.cl * model.cal_cl_loss([user_idx, pos_idx])
                        # rec_loss = model.bpr_loss(user_emb, pos_item_emb, neg_item_emb)   
                        batch_loss = l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                        b[0].append(batch_loss.item())
                        # b[1].append(rec_loss.item())
                        # Backward and optimize
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                        t.update(self.args.batch_size)
                        t.set_postfix(batch_loss=batch_loss.item(), cl_loss=cl_loss.item())
                        # , rec_loss=rec_loss.item(),best_epoch=self.best_epoch
                    if picture is not None:
                        self.flag=True
                        picture.update(np.mean(b,axis=1))

            # 调整阶段(训练NeuMF)
            
            optimizer = torch.optim.Adam([{'params':model.NeuMF.parameters(), 'weight+decay':self.args.reg},
                                        {'params':model.embedding_dict.parameters(), 'weight+decay':self.args.reg}], lr=self.args.lRate)
            for bpr_epoch in range(self.bprEpoch):
                if early_stop >= self.max_stop:
                    print(f"{emoji.emojize(':check_mark_button:')}{'-'*10}early stopping{'-'*10}{emoji.emojize(':check_mark_button:')}")
                    break
                with tqdm(total=len(self.data.training_data), desc=f"bprEpoch:{bpr_epoch+1}/{self.bprEpoch}", ncols=130) as t:
                    for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                        user_idx, pos_idx, neg_idx = batch
                        model.train()
                        rec_user_emb, rec_item_emb = model.info_forward()
                        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                        rec_loss = model.bce_loss(user_emb, pos_item_emb)
                        batch_loss = rec_loss
                        b[0].append(batch_loss.item())
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                        t.update(self.args.batch_size)
                        t.set_postfix(batch_loss=batch_loss.item(), rec_loss=rec_loss.item(), best_epoch = self.best_epoch)
                    if picture is not None:
                        self.flag=True
                        picture.update(np.mean(b,axis=1))
                    model.eval()
                    with torch.no_grad():
                        self.user_emb, self.item_emb = model.info_forward()
                    if bpr_epoch % evalNum == 0:
                        measure, flag = self.evaluate(bpr_epoch)
                        if flag:
                            early_stop = 0
                        else:
                            early_stop += 1
                        best = ':love_letter:' if flag else ""
                        t.write(f"{emoji.emojize(best)}Epoch:{bpr_epoch + 1}/{self.bprEpoch}.Evaluating The Model...\
                                {emoji.emojize(':face_with_monocle:')}\nRecommender Model:{self.args.model_name}")
                        for i in measure:
                            t.write(f"Top{i}:{measure[i]}")
        self.model.NeuMF = self.best_para
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        # self.measure, self.stds, self.targetHR = self.test(self.data.test_set, flag = True)
        self.measure, self.stds_all, self.stds_k, self.targetHR = self.test(self.data.test_set, flag = True)
        print(f"{emoji.emojize(':hundred_points:')} Test Performance {emoji.emojize(':hundred_points:')}")
        for i in self.measure:
            print(f"Top{i}:{self.measure[i]}")

    def save(self, epoch):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.info_forward()
            self.best_para = self.model.NeuMF
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

    def test(self, set_type, flag=False):
        # predict
        rec_list = {}
        stds_all = []
        stds_k = defaultdict(list)
        target_HR = np.zeros(len(self.topN))
        for i, user in enumerate(set_type):
            with torch.no_grad():
                user_emb = self.user_emb[self.data.get_user_id(user)].unsqueeze_(0).repeat(self.item_emb.shape[0],1)
                scores = self.model.NeuMF(user_emb, self.item_emb)
            candidates = scores.squeeze().cpu()
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
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
        self.NeuMF = NeuMF(embedding_size=emb_size)

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

    def info_forward(self, perturbed=False):
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
    
    def bpr_loss(self, user_emb, pos_item_emb, neg_item_meb):
        pos_ratings = self.NeuMF(user_emb, pos_item_emb)
        neg_ratings = self.NeuMF(user_emb, neg_item_meb)
        loss = -torch.log(10e-8 + torch.sigmoid(pos_ratings - neg_ratings))
        return torch.mean(loss)

    def bce_loss(self, user_emb, pos_item_emb):
        crit = nn.BCELoss()
        pos_ratings = self.NeuMF(user_emb, pos_item_emb)
        rate = torch.ones_like(pos_ratings).cuda()
        loss = crit(pos_ratings,rate)
        return loss 

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.info_forward(perturbed=True)
        user_view_2, item_view_2 = self.info_forward(perturbed=True)
        uu, ii = None, None
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2, ee=uu)
        return user_cl_loss

class NeuMF(nn.Module):
    def __init__(self, embedding_size):
        initializer = nn.init.xavier_uniform_
        super(NeuMF, self).__init__()
        layers = [2*embedding_size, 4*embedding_size, 8*embedding_size, 8*embedding_size, 4*embedding_size, 4*embedding_size, 4*embedding_size, 2*embedding_size, 2*embedding_size, embedding_size]
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_features=in_size, out_features=out_size))
        self.gmf = nn.Linear(in_features=embedding_size, out_features=layers[-1])
        self.affine_output = nn.Linear(in_features=2 * layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        for sm in self.modules():
            if isinstance(sm, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(sm.weight.data)
        self.cuda()
        
    def forward(self, user_emb, item_emb):
        mlp_emb = torch.cat([user_emb, item_emb],dim=-1)
        gmf_emb = torch.mul(user_emb, item_emb)
        gmf_emb = self.gmf(gmf_emb)
        for fc in self.fc_layers:
            mlp_emb = fc(mlp_emb)
            mlp_emb = nn.ReLU()(mlp_emb)
            # mlp_emb = nn.DropOut()(mlp_emb)
        emb = torch.cat([mlp_emb, gmf_emb], dim=-1)
        ratings = self.affine_output(emb)
        ratings = self.logistic(ratings)
        return ratings
      
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
