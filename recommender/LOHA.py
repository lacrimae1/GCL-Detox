import torch, emoji, random
import torch.nn as nn
import scipy.sparse as sp
import torch.sparse as tp
import torch.nn.functional as F
import numpy as np
from sympy.sets.sets import is_function_invertible_in_set
from scipy.sparse.linalg import eigs
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss, InfoNCE
from util.metrics import ranking_evaluation
from tqdm import tqdm
from collections import defaultdict
from util.tool import visualize

class LOHA():
    def __init__(self, args, data, attackName, method=[], W=None):
        self.data = data
        self.args = args
        self.bestPerformance = []
        self.recOutput = []
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.method = method
        self.ModelName = "LOHA"
        self.dataName = args.dataset
        self.attackName = attackName
        self.dataName_type = "_" + "_".join(method) if len(method)!=0 else ""
        if W is not None: 
            self.dataName_type += '_' + str(self.W)
        else:
            self.dataName_type += '_origin'
        self.K = 2
        self.drop_rate = 0.1
        self.cl = 0.1
        self.model = LOHA_Encoder(self.data, self.args.emb_size, self.drop_rate, self.K)
        self.best_epoch = 0
        self.flag = False
        self.max_stop = 40
    
    def add(self, targetItem, target):
        self.targetItem = targetItem
        self.target = target

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
        for epoch in range(self.maxEpoch):
            if early_stop >= self.max_stop and not is_first:
                print(f"{emoji.emojize(':check_mark_button:')}{'-'*10}Early stopping{'-'*10}{emoji.emojize(':check_mark_button:')}")
                break
            with tqdm(total=len(self.data.training_data), desc=f"Epoch:{epoch+1}/{self.maxEpoch}", ncols=130) as t:
                b = [[] for i in range(2)]
                if "ib" in self.method:
                    self.joint = False
                    if epoch <= self.W:
                        print(f"current epoch={epoch} self.W={self.W}")
                        self.bpr = 0
                        self.cl = 0.2
                        self.stage = True
                    else:
                        # if epoch == self.W+1:
                        #     visualize(self.model.embedding_dict["user_emb"], 
                        #         self.model.embedding_dict["item_emb"],
                        #         self.ModelName, self.attackName, self.dataName_type, self.target, attack=f"_{attack}", dataName=self.dataName)
                        self.stage = False
                        self.bpr = 1
                        self.cl = 0
                for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    model.train()
                    Z_l, Z_h, Z_f = model()
                    rec_user_emb, rec_item_emb = Z_f[:self.data.user_num,:], Z_f[self.data.user_num:,:]
                    self.optimizer.zero_grad()
                    cl_loss, rec_loss = model.loss([Z_l, Z_h, Z_f], [user_idx, pos_idx, neg_idx])
                    # Backward and optimize
                    cl_loss = cl_loss * self.cl
                    batch_loss = cl_loss + rec_loss
                    batch_loss.backward()

                    if requires_adjgrad and self.maxEpoch - epoch < gradIterationNum:
                        self.Matgrad += self.model.sparse_norm_adj.grad
                    elif requires_embgrad and self.maxEpoch - epoch < gradIterationNum:
                        self.usergrad += self.model.embedding_dict["user_emb"].grad
                        self.itemgrad += self.model.embedding_dict["item_emb"].grad

                    self.optimizer.step()
                    t.update(self.args.batch_size)
                    t.set_postfix(batch_loss=batch_loss.item(), rec_loss=rec_loss.item(),
                                cl_loss=cl_loss.item(),best_epoch=self.best_epoch)
                if picture is not None:
                    self.flag=True
                    picture.update(np.mean(b,axis=1),label=["batch","rec"])
                model.eval()
                with torch.no_grad():
                    _, _, Z_f = model()
                    self.user_emb, self.item_emb = Z_f[:self.data.user_num,:], Z_f[self.data.user_num:,:]
                if epoch % evalNum == 0:
                    if 'ib' in self.method:
                        if not self.stage:
                            measure, flag = self.evaluate(epoch)
                            if flag:
                                if epoch > self.W + 1:
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
        for i in self.measure:
            print(f"Top{i}:{self.measure[i]}")
        return saved

    def save(self, epoch):
        with torch.no_grad():
            self.best_emb_dict = self.model.embedding_dict
            _, _, Z_f = self.model()
            self.best_user_emb, self.best_item_emb = Z_f[:self.data.user_num,:], Z_f[self.data.user_num:,:]
            self.best_model = self
            if self.flag:self.best_epoch = epoch+1

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
        
class LOHA_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, K):
        super(LOHA_Encoder, self).__init__()
        self.data = data
        self.user_num = data.user_num
        self.emb_size = emb_size
        self.drop_rate = drop_rate
        self.T_k = self.chebyshev_polynomials(K=K)
        self.K = K
        self._init_model()
        self.filter_low = FilterGenerator(K, mode='low')
        self.filter_high = FilterGenerator(K, mode='high')
        self.mlp_LH = nn.Sequential(
            nn.Linear(emb_size,2*emb_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(2*emb_size, emb_size),
            nn.ReLU()
        )
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })

    def forward(self):
        X = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)  # [N_user+N_item, D]
        gamma_l = self.filter_low()
        gamma_h = self.filter_high()
        try:
            Z_l = torch.einsum('k,knm,md->nd', gamma_l, self.T_k, X)
            Z_h = torch.einsum('k,knm,md->nd', gamma_h, self.T_k, X)
        except:
            Z_l = torch.zeros_like(X)
            for k in range(self.K + 1):
                T_k_sparse = self.T_k[k].to_sparse()   # 如果 T_k 稠密
                Z_l += gamma_l[k] * torch.sparse.mm(T_k_sparse, X)       
            Z_h = torch.zeros_like(X)
            for k in range(self.K + 1):
                T_k_sparse = self.T_k[k].to_sparse()   # 如果 T_k 稠密
                Z_h += gamma_h[k] * torch.sparse.mm(T_k_sparse, X)  
        # Z_l = self.mlp_element(gamma_l, self.T_k, X)
        # Z_h = self.mlp_element(gamma_h, self.T_k, X)
        Z_l = self.mlp_LH(Z_l)
        Z_h = self.mlp_LH(Z_h)
        Z_f = 0.8*Z_l + 0.2*Z_h
        return Z_l, Z_h, Z_f

    def mlp_element(self, gamma, T_k, X):
        j = np.arange(self.K + 1)
        xj = np.cos((j+0.5)/(self.K+1)*np.pi)
        T = []
        for j in range(self.K+1):
            T_j = [1, xj[j]]
            for k in range(2,self.K+1):
                t = 2*xj[j]*T_j[-1] - T_j[-2]
                T_j.append(t)
            T.append(T_j)
        res = torch.zeros_like(X)
        for k in range(self.K+1):
            wk = 0
            for j in range(self.K+1):
                wk += gamma[j] * T[j][k]
            wk = 2*wk / (self.K+1)
            res += wk * T_k[k] @ X
        return res

    def loss(self, emb, idx ,tau=0.5, mu=1):
        Z_l, Z_h, Z_f = emb
        Z_l_u, Z_l_i, Z_h_u, Z_h_i, Z_f_u, Z_f_i = Z_l[:self.user_num,:], Z_l[self.user_num:,:], Z_h[:self.user_num,:], Z_h[self.user_num:,:], Z_f[:self.user_num,:], Z_f[self.user_num:,:]
        user_idx, pos_idx, neg_idx = idx
        user_emb, pos_emb, neg_emb = Z_f_u[user_idx], Z_f_i[pos_idx], Z_f_i[neg_idx]
        pos_score = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_emb).sum(dim=1)
        bpr_loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).mean()

        # 计算C-(zl,zh)
        trend_l = self._spectral_trend(Z_l, self.L_sym)
        trend_h = self._spectral_trend(Z_h, self.L_sym)
        C_minus = torch.abs(trend_l - trend_h)

        # Eq.(12) 的 InfoNCE 形式
        Ll_u, Lh_u = self.loss_LH(Z_f_u[user_idx], Z_l_u[user_idx], Z_h_u[user_idx], tau)
        Ll_i, Lh_i = self.loss_LH(Z_f_i[pos_idx], Z_l_i[pos_idx], Z_h_i[pos_idx], tau)

        # Eq.(13) 的 Lsf
        X = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        Lsf = self.loss_sf(Z_f, X, self.L_sym, tau)
        return Ll_u+Lh_u+Ll_i+Lh_i+mu*Lsf, bpr_loss
    
    def loss_LH(self, Z_f, Z_l, Z_h, tau):
        device = Z_f.device
        N = Z_f.size(0)
        # 归一化
        Z_f = F.normalize(Z_f, dim=-1)
        Z_l = F.normalize(Z_l, dim=-1)
        Z_h = F.normalize(Z_h, dim=-1)

        # 分子：zf_i 与 zl_i / zh_i 的相似度
        sim_zf_zl = torch.exp((Z_f * Z_l).sum(-1) / tau)   # [N]
        sim_zf_zh = torch.exp((Z_f * Z_h).sum(-1) / tau)   # [N]

        # 负样本：zf_i 与所有其他节点的 zl_p / zh_p
        logits_zf_zl = torch.exp(torch.mm(Z_f, Z_l.t()) / tau)  # [N,N]
        logits_zf_zh = torch.exp(torch.mm(Z_f, Z_h.t()) / tau)  # [N,N]

        # 构造掩码排除对角线
        mask = torch.eye(N, device=device).bool()
        logits_zf_zl = logits_zf_zl.masked_fill(mask, 0.0)
        logits_zf_zh = logits_zf_zh.masked_fill(mask, 0.0)

        # 分母：负样本 + zl_i 与 zh_i 的额外负对
        neg_sum_zl = logits_zf_zl.sum(1) + torch.exp((Z_l * Z_h).sum(-1) / tau)
        neg_sum_zh = logits_zf_zh.sum(1) + torch.exp((Z_l * Z_h).sum(-1) / tau)

        # 逐节点 Info-NCE
        Ll = -torch.log(sim_zf_zl / (neg_sum_zl + 1e-8)).mean()
        Lh = -torch.log(sim_zf_zh / (neg_sum_zh + 1e-8)).mean()
        return Ll, Lh
    
    def loss_sf(self, Z_f, X, L_norm, tau):
        N = Z_f.size(0)
        device = Z_f.device

        # 1. 计算论文 Eq.(5) 的 Tr(x_i)
        Tr = self._spectral_trend(X, L_norm)

        # 2. C- = ||Tr||_2  (论文把 Tr 作为标量，直接取绝对值即可)
        C_minus = Tr.abs()

        # 3. Info-NCE
        Z_f = F.normalize(Z_f, dim=-1)
        C_minus = F.normalize(C_minus, dim=0)
        def block_mm(A, B, block_size=1024):
            M, K = A.shape
            K, N = B.shape
            C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
            for i in range(0, M, block_size):
                for j in range(0, N, block_size):
                    end_i = min(i + block_size, M)
                    end_j = min(j + block_size, N)
                    C[i:end_i, j:end_j] = A[i:end_i, :] @ B[:, j:end_j]
            return C
        pos_sim = torch.exp((Z_f * C_minus.squeeze(1)).sum(-1) / tau)
        try:
            logits = torch.exp((Z_f @ C_minus.T).squeeze(-1) / tau)
        except:
            torch.cuda.empty_cache()
            logits = block_mm(Z_f.half(), C_minus.T.half())
        mask = torch.eye(N, device=device).bool()
        logits = logits.masked_fill(mask, 0.0)
        neg_sum = logits.sum(1) + pos_sim

        L_sf = -torch.log(pos_sim / (neg_sum + 1e-8)).mean()
        return L_sf

    def _sim(self, z1, z2, tau):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.exp(torch.sum(z1*z2, dim=-1) / tau)

    def _spectral_trend(self, X, L_norm):
        deg_sparse = tp.sum(L_norm, dim=1)
        deg = deg_sparse.to_dense().float().clamp(min=1).to("cuda")
        X_norm = X / deg.sqrt().unsqueeze(-1)
        neighbor_mean = torch.spmm(L_norm, X_norm) / deg.unsqueeze(-1)
        trend = (X_norm - neighbor_mean).pow(2)
        return trend
        
    def chebyshev_polynomials(self, K:int)->torch.Tensor:
        self.adj = self.data.ui_adj
        d = np.asarray(self.adj.sum(1)).ravel().astype(float)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt, 0, format='csr')

        I = sp.eye(self.adj.shape[0], format='csr')
        L_sym = I - D_inv_sqrt @ self.adj @ D_inv_sqrt
        # 转变为pytorch稀疏张量
        L_sym = L_sym.tocoo().astype(float)
        indices = torch.tensor([L_sym.row, L_sym.col], dtype=torch.long)
        values = torch.tensor(L_sym.data, dtype = torch.float32)
        shape = torch.Size(L_sym.shape)
        L_sym = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        self.L_sym = L_sym.to("cuda")
        # 转变为pytorch稠密张量
        # L_sym = torch.from_numpy(L_sym.todense()).float()

        # 公式8的chebyshev展开
        L_scipy = sp.coo_matrix((L_sym.values().cpu().numpy(),
                (L_sym.indices()[0].cpu().numpy(), L_sym.indices()[1].cpu().numpy())),
                shape=L_sym.shape)
        vals = eigs(L_scipy, k=1, which='LR', return_eigenvectors=False)
        lambda_max = float(vals.real.max())
        L_sym = L_sym.to_dense()
        L_scaled = 2*L_sym/lambda_max - torch.eye(L_sym.size(0))
        T = [torch.eye(L_sym.size(0)), L_scaled]
        for k in range(2, K+1):
            T_k = 2*L_scaled @ T[-1] - T[-2]
            T.append(T_k)
        return torch.stack(T, dim=0).to("cuda") #[K+1,N,N] N为节点数量

class FilterGenerator(nn.Module):
    """
    生成低通/高通滤波器的 learnable γ 参数
    论文 Eq.(9) 的完整实现
    """
    def __init__(self, K: int, mode: str = 'low'):
        super().__init__()
        assert mode in {'low', 'high'}
        self.K = K
        self.mode = mode
        # β_a, β_b, δ 均为可训练参数
        self.beta_a = nn.Parameter(torch.tensor(0.0 if mode == 'low' else 2.0),requires_grad=True)
        self.beta_b = nn.Parameter(torch.tensor(2.0),requires_grad=True)
        self.delta = nn.Parameter(torch.tensor(0.0),requires_grad=True)

    def forward(self) -> torch.Tensor:
        self.device = self.delta.device
        j = torch.arange(self.K + 1, dtype=torch.float32, device=self.device)
        cos_term = 1 + torch.cos(1 + 0.5 * torch.tanh(self.delta) + j / self.K * torch.pi)
        if self.mode == 'low':
            gamma = F.relu(self.beta_a) - 0.5 * F.relu(self.beta_b) * cos_term
        else:
            gamma = F.relu(self.beta_a) + 0.5 * F.relu(self.beta_b) * cos_term
        return gamma  # [K+1]