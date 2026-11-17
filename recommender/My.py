import torch, emoji, random
import torch.nn as nn
import scipy.sparse as sp
import torch.sparse as tp
import torch.nn.functional as F
import numpy as np
from sympy.sets.sets import is_function_invertible_in_set
from scipy.sparse.linalg import eigsh
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss, InfoNCE
from util.metrics import ranking_evaluation
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.utils import dense_to_sparse, remove_self_loops, to_dense_adj
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from torch.autograd import Function
from torch_scatter import scatter_add
from tqdm import tqdm
from collections import defaultdict
from util.tool import visualize

class My():
    def __init__(self, args, data, attackName, method=[], W=None):
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
        # -n_layer 2 -lambda 0.5 -eps 0.1
        self.max_stop = 20 if len(self.method)==0 else 40
        self.n_layers = 2
        self.temp = 0.2
        self.cl = 0.2
        self.bpr = 1
        self.stage = False
        self.joint = True
        self.eps = 0.1
        self.model = My_Encoder(self.data, self.args.emb_size, self.eps, self.n_layers, self.temp)
        self.flag=False
        self.W = W
    
    def add(self, targetItem, target):
        self.targetItem = targetItem
        self.target = target

    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None,
               evalNum=5,picture=None, attack = 0):
        self.bestPerformance=[]
        model = self.model.cuda()
        if optimizer is None: 
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
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
                    rec_user_emb, rec_item_emb = model(model.edge_index,model.adj)
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
                    if not self.joint:
                        if self.stage:
                            cl_loss = self.cl * model.cal_cl_loss([user_idx, pos_idx])
                            rec_loss = torch.tensor(0).cuda()
                        else:
                            rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                            cl_loss = torch.tensor(0).cuda()
                    else:
                        cl_loss = self.cl * model.cal_cl_loss([user_idx, pos_idx])
                        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                    batch_loss = rec_loss + l2_reg_loss(self.args.reg, user_emb, pos_item_emb) + cl_loss
                    b[0].append(batch_loss.item())
                    b[1].append(rec_loss.item())
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
                                cl_loss=cl_loss.item(),best_epoch=self.best_epoch)
                if picture is not None:
                    self.flag=True
                    picture.update(np.mean(b,axis=1),label=["batch","rec"])
                model.eval()
                with torch.no_grad():
                    self.user_emb, self.item_emb = model(model.edge_index, model.adj)
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
            self.best_user_emb, self.best_item_emb = self.model(self.model.edge_index, self.model.adj)
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
        
class My_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, convs=None):
        super(My_Encoder, self).__init__()
        self.data = data
        self.user_num = data.user_num
        self.emb_size = emb_size
        self.drop_rate = drop_rate
        self.n_layers = n_layers
        self.temp = temp
        self.num_nodes = data.user_num + data.item_num
        self.adj = data.ui_adj # 得到原始的邻接矩阵
        self.edge_index = self.trans_edge_index(self.adj) # 原始图的edge_index
        self._init_model()
        self.build_encoder(emb_size, 2*emb_size, emb_size)

    def build_encoder(self,in_dim, hidden_dim, out_dim, num_layers=2, diff_layers=2):
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def _init_uiAdj(self, ui_adj):
        self.sparse_norm_adj = sp.diags(np.array((1 / np.sqrt(ui_adj.sum(1)))).flatten()) @ ui_adj @ sp.diags(
            np.array((1 / np.sqrt(ui_adj.sum(0)))).flatten())
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.sparse_norm_adj).cuda()

    def forward(self, edge_index, adj=None, diff_layer=0):
        x = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = adj * deg_inv_sqrt.view(-1,1)
        adj = adj * deg_inv_sqrt.view(1,-1)
        def diffusion(H):
            return adj @ H 
        for i, conv in enumerate(self.convs):
            x = diffusion(x)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        for _ in range(diff_layer):
            x = diffusion(x)
        user_all_embeddings, item_all_embeddings = torch.split(x, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(self.edge1, adj=self.adj1)
        user_view_2, item_view_2 = self.forward(self.edge2, adj=self.adj2, diff_layer=2)
        def upper_lower_bound_loss(h1, h2, alpha=5.0, beta=9.0):
            pos_dist = torch.norm(h1-h2, dim=1)
            neg_dist = torch.norm(h1-h2.roll(1,dims=0),dim=1)
            lower = torch.relu(pos_dist-neg_dist+alpha).sum()
            upper = torch.relu(neg_dist-pos_dist-beta).sum()
            return lower+upper
        return InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp) + \
                InfoNCE(item_view_1[i_idx], item_view_2[i_idx],self.temp) +\
                upper_lower_bound_loss(user_view_1, user_view_2) +\
                upper_lower_bound_loss(item_view_1, item_view_2)

    def trans_edge_index(self, adj):
        # 将原始的邻接矩阵转换成edge_index
        adj_coo = adj.tocoo()
        edge_index = torch.vstack([torch.from_numpy(adj_coo.row),torch.from_numpy(adj_coo.col)])
        return edge_index.long().cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        perturb1 = SoftSparseAdjPerturb(self.edge_index,num_nodes=self.num_nodes)
        perturb2 = SoftSparseAdjPerturb(self.edge_index,num_nodes=self.num_nodes)
        self.edge1 = perturb1.optimize()
        self.edge2 = perturb2.optimize()
        # self.edge1 = (self.spectral_augmentation(self.edge_index, torch.rand(self.num_nodes, self.num_nodes)*0.2))
        # self.edge2 = (self.spectral_augmentation(self.edge_index, torch.rand(self.num_nodes, self.num_nodes)*0.2))
        def trans(edge):
            row, col = edge
            adj = SparseTensor(row=row, col=col, value=torch.ones(row.size(0), device=edge.device),
                        sparse_sizes=(self.num_nodes, self.num_nodes))
            return adj
        self.adj = trans(self.edge_index)
        self.adj1 = trans(self.edge1)
        self.adj2 = trans(self.edge2)
    
    # def get_normalized_laplacian(self, edge_index):
    #     # assert edge_index.max() < self.num_nodes, "edge_index 中的节点索引超出了 num_nodes"
    #     edge_index, edge_weight = get_laplacian(edge_index, normalization='sym',num_nodes=self.num_nodes)
    #     # print(f"edge_index.shape[1]={edge_index.shape[1]}\nedge_weight.shape[0]={edge_weight.shape[0]}")
    #     L = to_scipy_sparse_matrix(edge_index, edge_weight, self.num_nodes).tocsc()
    #     return L
    
    # def laplacian_eigenvalues(self, L, k=10):
    #     eigenvalues, _ = eigsh(L, k=k, which='SM')
    #     return torch.tensor(eigenvalues, dtype=torch.float32)

    # def bernoulli_perturbation(self, A, delta, eps=0.1):
    #     # 生成扰动的邻接矩阵,使用bernoulli采样
    #     M = torch.bernoulli(delta).bool()
    #     A_perturbed = A.clone()
    #     A_perturbed[M] = 1-A_perturbed[M] # flip edges
    #     return A_perturbed

    # def spectral_augmentation(self, edge_index, delta, eps=0.1, lr=1e-3, steps=20):
    #     # 优化delta以最小化频谱差异
    #     A = torch.zeros(self.num_nodes, self.num_nodes)
    #     A[edge_index[0], edge_index[1]] = 1
    #     A = A + A.t()  # undirected
    #     L_orig = self.get_normalized_laplacian(edge_index)
    #     eig_orig = self.laplacian_eigenvalues(L_orig)

    #     delta = delta.clone().detach().requires_grad_(True)
    #     optimizer = torch.optim.Adam([delta], lr=lr)

    #     for _ in range(steps):
    #         A_pert = self.bernoulli_perturbation(A, delta, eps)
    #         edge_pert = A_pert.nonzero().t()
    #         L_pert = self.get_normalized_laplacian(edge_pert)
    #         eig_pert = self.laplacian_eigenvalues(L_pert)
    #         loss = F.mse_loss(eig_pert, eig_orig)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         delta.data.clamp_(0, 1)

    #     with torch.no_grad():
    #         A_final = self.bernoulli_perturbation(A, delta, eps)
    #         edge_final = A_final.nonzero().t()
    #     return edge_final.cuda()
    
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)    

class SoftSparseAdjPerturb(nn.Module):
    def __init__(self, edge_index, num_nodes, epsilon=0.2, lr=0.01, steps=50, device='cuda'):
        super().__init__()
        self.num_nodes = num_nodes
        self.epsilon = epsilon
        self.steps = steps
        self.device = device

        edge_index, _ = remove_self_loops(edge_index)
        self.register_buffer('A_orig', to_dense_adj(edge_index, max_num_nodes=num_nodes)[0])
        with torch.no_grad():
            self.A_orig = self.A_orig.cuda()
            A_orig_plus_I = self.A_orig + torch.eye(self.num_nodes, device=self.device)
            self.eig_orig = self._eigvals_fast(A_orig_plus_I)
        self.candidate_edges = self._sample_missing_edges()
        self.K = self.candidate_edges.size(1)
        self.delta = nn.Parameter(torch.zeros(self.K, device=device))
        self.opt = torch.optim.Adam([self.delta], lr=lr)

    def _sample_missing_edges(self):
        A = self.A_orig
        mask = (A == 0) & (A.t() == 0)
        row, col = torch.where(mask)
        K = max(1, int(self.epsilon * self.num_nodes * self.num_nodes))
        perm = torch.randperm(row.size(0))[:K]
        return torch.stack([row[perm], col[perm]], dim=0)

    def _soft_adj(self):
        w = torch.sigmoid(self.delta)
        adj = torch.sparse_coo_tensor(
            self.candidate_edges, w,
            size=(self.num_nodes, self.num_nodes),
            device=self.device
        ).coalesce().to_dense()
        adj = adj + adj.t()
        return self.A_orig + adj

    def _eigvals_fast(self, A):
        k = min(16, self.num_nodes - 1)
        edge_idx, _ = dense_to_sparse(A)
        row, col = edge_idx[0], edge_idx[1]
        val = torch.ones_like(row, dtype=torch.float32, device=self.device, requires_grad=True)
        return autograd_power_eig(val, row, col, self.num_nodes, k)

    def _step(self):
        A_soft = self._soft_adj()
        eig_pert = self._eigvals_fast(A_soft)
        loss = torch.norm(self.eig_orig - eig_pert, p=2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def optimize(self, verbose=False):
        self.train()
        for i in range(self.steps):
            loss = self._step()
            if i % 10 == 0:
                print(f'step {i}, loss={loss:.4f}')
        self.eval()
        with torch.no_grad():
            A_final = self._soft_adj()
            mask = (A_final>=0.5).float()
            edge_index, _  = dense_to_sparse(mask)
        return edge_index.long()

class AutogradPowerEig(Function):
    """
    可微幂迭代：只返回前 k 个最小特征值
    输入：val（需要梯度）、row、col、N
    输出：Tensor[k]（带 grad_fn）
    """

    @staticmethod
    def forward(ctx, val, row, col, N, k=16, max_iter=40, tol=1e-5):
        device = val.device
        # 构造 L = I - D^{-1/2} A D^{-1/2} 的乘法算子
        deg = torch.zeros(N, device=device).index_add(0, row, val).index_add(0, col, val)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        y_row = -deg_inv_sqrt[row] * val * deg_inv_sqrt[col]  # [E]

        # 前向：幂迭代 + Rayleigh
        V = torch.randn(N, k, device=device)
        V, _ = torch.linalg.qr(V)
        for _ in range(max_iter):
            # L@V
            contrib = y_row.unsqueeze(1) * V[col, :]
            LV = scatter_add(contrib, row, dim=0, dim_size=N)
            contrib = y_row.unsqueeze(1) * V[row, :]
            LV += scatter_add(contrib, col, dim=0, dim_size=N)
            LV += V  # +I
            V_new, _ = torch.linalg.qr(LV)
            if torch.norm(V_new - V, p='fro') < tol:
                break
            V = V_new
        rayleigh = (V * LV).sum(0)  # [k]
        ctx.save_for_backward(val, row, col, V, rayleigh)
        ctx.N, ctx.k, ctx.y_row = N, k, y_row
        return rayleigh

    @staticmethod
    def backward(ctx, grad_output):
        # 简化梯度：d loss / d val ≈ ∑_i grad_output[i] * (v_u - v_v)^2 * |weight|
        val, row, col, V, rayleigh = ctx.saved_tensors
        N, k, y_row = ctx.N, ctx.k, ctx.y_row
        grad_val = torch.zeros_like(val)
        for i in range(k):
            g = grad_output[i]
            contrib = g * (V[row, i] - V[col, i]) ** 2
            grad_val += contrib * y_row.abs()  # 近似梯度
        return grad_val, None, None, None, None, None, None


def autograd_power_eig(val, row, col, N, k=16):
    return AutogradPowerEig.apply(val, row, col, N, k)