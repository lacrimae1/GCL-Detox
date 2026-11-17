import torch
import torch.nn.functional as F
import numpy as np

class RobustLossComputer:
    """
    只负责计算三阶段损失，返回张量
    外部自己:loss.backward() / optimizer.step()
    """

    def __init__(self, model, info_nce_fn, bpr_fn, l2_loss, reg=0.2, device='cuda'):
        self.model       = model.to(device)
        self.info_nce_fn = info_nce_fn
        self.bpr_fn      = bpr_fn
        self.device      = device
        self.l2_loss     = l2_loss
        self.reg = reg
        self.cl = 0.2
        self.f_anchor    = None          # 结构锚点
        self.flag = False
        self.stage       = 1             # 当前阶段 1/2/3

    # ---------------- 外部唯一入口 ----------------
    def compute(self, batch, epoch_args):
        """
        epoch_args:  dict 必须含
           'stage'   : 1 or 2 or 3
           'lambda'  : float  阶段2专用
           'gamma'   : float  阶段3专用
           'alpha'   : float  阶段2/3结构保留强度
        返回:  一个标量张量（已放在 device 上）
        """
        self.stage = epoch_args['stage']
        if self.stage == 1:
            return self._loss_stage1(batch)
        if self.stage == 2:
            return self._loss_stage2(batch, epoch_args['lambda'], epoch_args['alpha'])
        if self.stage == 3:
            return self._loss_stage3(batch, epoch_args['gamma'], epoch_args['alpha'])
        raise ValueError('stage must be 1/2/3')

    # ---------------- 阶段1：纯 InfoNCE ----------------
    def _loss_stage1(self, batch):
        user_idx, pos_idx, neg_idx = batch
        rec_user_emb, rec_item_emb = self.model()
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
        # u_idx = torch.unique(torch.Tensor(batch[0]).type(torch.long)).cuda()
        # i_idx = torch.unique(torch.Tensor(batch[1]).type(torch.long)).cuda()
        # user_view_1, item_view_1 = self.model(perturbed=True)
        # user_view_2, item_view_2 = self.model(perturbed=True)
        # user_cl_loss = self.info_nce_fn(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        # item_cl_loss = self.info_nce_fn(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        cl_loss = self.model.cal_cl_loss([user_idx, pos_idx])
        loss = cl_loss + self.l2_loss(self.reg,user_emb,pos_item_emb)
        return self.cl*loss, torch.tensor(0).cuda(), self.cl*loss

    # ---------------- 阶段2：渐进 ----------------
    def _loss_stage2(self, batch, λ, alpha):
        user_idx, pos_idx, neg_idx = batch
        rec_user_emb, rec_item_emb = self.model()
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
        cl_loss = self.model.cal_cl_loss([user_idx, pos_idx])
        loss_bpr = self.bpr_fn(user_emb, pos_item_emb, neg_item_emb)
        loss = (1 - λ) * loss_cl + λ * loss_bpr
        if self.f_anchor is not None:
            loss += alpha * F.mse_loss(user_emb, self.f_anchor[user_idx])
        return loss+ self.l2_loss(self.reg,user_emb,pos_item_emb), loss_bpr, loss_cl

    # ---------------- 阶段3：纯 BPR + 护栏 ----------------
    def _loss_stage3(self, batch, gamma, alpha):
        user_idx, pos_idx, neg_idx = batch
        rec_user_emb, rec_item_emb = self.model()
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
        loss_bpr = self.bpr_fn(user_emb, pos_item_emb, neg_item_emb)

        # 梯度护栏：先算梯度范数（不开反向图）
        loss_bpr.backward(retain_graph=True)
        grad_norm = self._grad_norm()

        if grad_norm > gamma:        # 触发护栏
            loss_bpr += alpha * F.mse_loss(user_emb, self.f_anchor[user_idx])
        return loss_bpr + self.l2_loss(self.reg,user_emb,pos_item_emb), loss_bpr, torch.tensor(0).cuda()

    # ---------------- 工具 ----------------
    def _grad_norm(self):
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def set_anchor(self):
        """外部在阶段2开始前调用:把对比阶段最后一轮 user emb 存下来"""
        if not self.flag:
            self.flag = True
            with torch.no_grad():
                user_emb, _ = self.model()
                self.f_anchor = user_emb.detach().clone()
