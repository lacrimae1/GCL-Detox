import numpy as np
from collections import defaultdict
from util.FileIO import FileIO
import scipy.sparse as sp
import torch

class DataLoader():
    def __init__(self, args,method, path=None, pathvt=None):
        if path == None:
            self.training_data = FileIO.load_data_set(args.data_path + args.dataset + args.training_data)
            self.val_data = FileIO.load_data_set(args.data_path + args.dataset + args.val_data)
            self.test_data = FileIO.load_data_set(args.data_path + args.dataset + args.test_data)
        else:
            print(f"load_path={path}")
            self.training_data = FileIO.load_data_set(path)
            self.val_data = FileIO.load_data_set(pathvt + args.val_data)
            self.test_data = FileIO.load_data_set(pathvt + args.test_data)

        self.dataName=args.dataset
        self.fakeNum = 0
        self.user = {}    # user2id
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.val_set = defaultdict(dict)
        self.val_set_item = set()
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.old = self.user_num
        self.item_num = len(self.training_set_i)
        if "lap" in method:self.W()
        self.ui_adj = self.__create_sparse_bipartite_adjacency() # 邻接矩阵
        self.norm_adj = self.normalize_graph_mat(self.ui_adj) # 对称归一化后的邻接矩阵
        self.interaction_mat = self.__create_sparse_interaction_matrix() # user-item交互矩阵
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])

    def randomly_sample(self, sample_type=None):
        '''
        sampel_type:采样策略
        0: 选择超过200个交互的热门项目,然后随机采样与这些热门项目有交互的用户5000个
        1: 随机采样2000个用户
        2: 根据流行度,随机从top 5%中采样500个hot项目,从bottom 80%中采样500个cold项目.
            随机采样1000个用户in the same way.
        '''
        rng = np.random.default_rng(1)
        if sample_type == 0:
            size = int(((self.interaction_mat.shape[0] * 0.15) // 10)*10)
            col_sum = np.asarray(self.interaction_mat.sum(axis=0)).ravel()
            target_cols = np.where(col_sum>200)[0]
            if len(target_cols) == 0:
                raise ValueError("没有任何列之和 > 200")
            X_sub = self.interaction_mat[:, target_cols]
            row_idx, _ = X_sub.nonzero()
            unique_rows = np.unique(row_idx)
            size = min(size, len(unique_rows))
            sampled_rows = rng.choice(unique_rows, size = size, replace=False)
            return sampled_rows
        elif sample_type == 1:
            users = np.arange(self.user_item)
            return rng.choice(users, size=2000, replace=False)
        elif sample_type == 2:
            size = int(((self.interaction_mat.shape[0] * 0.5) // 10) * 10)
            user_sum = np.asarray(self.interaction_mat.sum(axis=1)).ravel()
            user_sorted = np.argsort(user_sum)
            sampled_user = user_sorted[:size]
            return sampled_user
        else:
            item_sum = np.asarray(self.interaction_mat.sum(axis=0)).ravel()
            item_sorted = np.argsort(item_sum)
            n_total = item_sum.size
            start, end = int(0.95*n_total), n_total
            candidates = item_sorted[start:end]
            hot_item = rng.choice(candidates, size=min(500, end-start), replace=False)

            start, end = 0, int(n_total*0.8)
            candidates = item_sorted[start:end]
            cold_item = rng.choice(candidates, size=min(500, end-start), replace=False)

            user_sum = np.asarray(self.interaction_mat.sum(axis=1)).ravel()
            user_sorted = np.argsort(user_sum)
            n_total = user_sum.size
            start, end = int(0.95*n_total), n_total
            candidates = user_sorted[start:end]
            hot_user = rng.choice(candidates, size=min(500, end-start), replace=False)

            start, end = 0, int(n_total*0.8)
            candidates = user_sorted[start:end]
            cold_user = rng.choice(candidates, size=min(500, end-start), replace=False)
            return hot_user, cold_user, hot_item, cold_item

    def invert(self, fakeName, action):
        self.fakeNum += 1
        self.training_data.extend([[fakeName, self.id2item[action], 1]])

    def reset(self):
        if self.fakeNum:
            self.training_data = self.training_data[:-self.fakeNum]
        self.interaction_mat[self.old:,:] = 0
        self.fakeNum = 0

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.val_data:
            user, item, rating = entry
            if user not in self.user:
                continue
            self.val_set[user][item] = rating
            self.val_set_item.add(item)
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)
    
    def category(self):
        pop = np.argsort(self.interaction_mat[:,:].sum(0))[0,:].tolist()[0]
        pop_num = (int)(self.item_num * 0.05)
        unpop_num = (int)(self.item_num*0.8)
        self.pop = pop[-pop_num:]
        self.unpop = pop[:unpop_num]
        self.normal = pop[unpop_num:pop_num]

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def create_sparse_matrix(self, data=None):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in data:
            if pair[1] in self.item and pair[0] in self.user:
                row += [self.user[pair[0]]]
                col += [self.item[pair[1]]]
                entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def val_size(self):
        return len(self.val_set), len(self.val_set_item), len(self.val_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated_id(self, u):
        return [self.item[i] for i in self.training_set_u[u].keys()]

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = self.__create_sparse_interaction_matrix()
        return m
    
    def W(self):
        res = torch.zeros((self.item_num,self.item_num))
        res += torch.eye(self.item_num)
        for i in range(self.item_num):
            one = set(self.training_set_i[self.id2item[i]].keys())
            for j in range(i+1, self.item_num):
                two = set(self.training_set_i[self.id2item[j]].keys())
                res[i][j] = len(one & two) / len(one | two)
        res += res.clone().T
        res -= torch.eye(self.item_num)
        self.res = res.cuda()

    def laplacian_W(self, item_index):
        return self.res[item_index,:][:,item_index]
