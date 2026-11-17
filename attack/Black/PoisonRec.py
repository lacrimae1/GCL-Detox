import os
import time
from copy import deepcopy
import gym
from gym import spaces
from gym.envs.registration import register
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli,Multinomial, Categorical
from stable_baselines3 import PPO
import numpy as np
import scipy.sparse as sp
import random
import matplotlib.pyplot as plt
from util.tool import targetItemSelect, dataSave
from util.metrics import AttackMetric
from scipy.sparse import vstack, csr_matrix
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy, MultiInputActorCriticPolicy, partial
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import PyTorchObs


class PoisonRec():
    def __init__(self, arg, data):
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]

        self.targetItem = targetItemSelect(data, arg) #物品编号
        self.targetItem = [data.item[i.strip()] for i in self.targetItem] #目标物品在邻接矩阵中的序号
        # [3378, 2706, 2246, 3234, 3476]
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # limitation 
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.env = None
        self.item_num = data.item_num
        #虚假用户在邻接矩阵中的序号
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

    def posionDataAttack(self,recommender):
        self.recommender = recommender
        self.fakeUserInject(self.recommender)
        # # if self.env is None:
        # policy_kwargs = dict(features_extractor_class=CustomFeaturesExtractor,
        #                      features_extractor_kwargs=dict(total_actions=self.maliciousFeedbackNum))
        policy_kwargs = dict(targetItem=self.targetItem)
        self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
        self.agent = PPO(CustomPolicy, self.env, verbose=1, batch_size=self.maliciousFeedbackNum,clip_range=0.1,
                          gamma=0.9,n_steps=3*self.maliciousFeedbackNum,n_epochs=20,policy_kwargs=policy_kwargs)
        self.agent.learn(total_timesteps=460000, callback=RewardCallback(self.recommender.model_name))
        # self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
        obs = self.env.reset()
        done = False
        total_reward = 0
        uiAdj = self.recommender.data.matrix()
        self.env.mode()
        while not self.env.fakeUserDone:
            uiAdj[self.fakeUser[self.env.fakeUserid],:] = 0
            while not done:
                action, _states = self.agent.predict(obs, deterministic=True)
                uiAdj[self.fakeUser[self.env.fakeUserid],action] = 1
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            uiAdj[self.fakeUser[self.env.fakeUserid], self.env.itemList] = 1
        self.interact = uiAdj
        return self.interact

    def fakeUserInject(self, recommender):
        Pu, Pi = recommender.model()
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser_{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser_{}".format(i)
        
        # 已经先注入含有目标物品的虚假用户
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem =  self.targetItem
            for i in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u],recommender.data.id2item[i]))
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]

        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)

        recommender.__init__(recommender.args, recommender.data)
        with torch.no_grad():
            try:
                recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
                recommender.model.embedding_dict['item_emb'][:] = Pi
                recommender.user_emb = recommender.model.embedding_dict['user_emb']
                recommender.item_emb = recommender.model.embedding_dict['item_emb'] 
            except:
                recommender.model.embedding_dict['user_mf_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], :Pu.shape[1]//2]
                recommender.model.embedding_dict['user_mlp_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], Pu.shape[1]//2:]
                recommender.model.embedding_dict['item_mf_emb'][:] = Pi[:, :Pi.shape[1]//2]
                recommender.model.embedding_dict['item_mlp_emb'][:] = Pi[:, Pi.shape[1]//2:]

        recommender.model = recommender.model.cuda()

class RewardCallback(BaseCallback):
    def __init__(self, recommenderName, verbose = 0):
        super(RewardCallback, self).__init__(verbose)
        self.recommenderName = recommenderName
        self.rewards = []  # 保存所有虚假用户的奖励
        self.max_reward = []  # 作图：保存虚假用户之间做大的奖励
        self.total_episode_rewards = []
        self.time = 0
        self.last_reward = []
        self.means = []
        self.stds = []
        self.path = "./result_show/PoisonRec/{}/".format(recommenderName) 
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.fig, self.ax = plt.subplots(2,1)
    
    def _on_step(self):
        return True
    def _on_rollout_end(self):
        pass
        # return super()._on_rollout_end()
    def end_episode(self):
        if self.locals.get('rewards') is not None:
            self.rewards.append(self.locals['rewards'][0])
            # self.current_episode_reward += self.locals['rewards'][0]
            # self.rewards.append(self.locals['rewards'][0])
        
    
    def end_total(self):
        self.time += 1
        # episode_num = self.locals["total_timesteps"] // self.locals["n_rollout_steps"] + 5
        self.means.append(np.mean(np.array(self.rewards)))
        self.stds.append(np.std(np.array(self.rewards)))
        self.total_episode_rewards.append(sum(self.rewards))
        self.max_reward.append(max(self.rewards))
        self.last_reward.append(self.rewards[-1])
        time = range(1, self.time + 1)
        
        xlim = [1, self.time + 10]
        if self.time < 50:
            xticks = range(1, self.time+1,5)
        else:
            xticks = range(1, self.time+1,10)
        # if self.time < episode_num//2:
        #     xlim = [1, self.time*2]
        #     xticks = range(1, self.time+1, 5)
        #     # np.append(xticks, self.time*2)
        # else:
        #     xlim = [1, episode_num]
        #     xticks = range(1, self.time+1, 10)
            # np.append(xticks, episode_num)
        # 重置变量
        # self.current_episode_reward = 0  # 重置当前episode的奖励
        self.rewards = []


        # 动态绘制奖励曲线        
        self.ax[0].clear()
        self.ax[0].plot(time, self.means, color='deeppink', label="mean")
        self.ax[0].fill_between(time, np.array(self.means)-np.array(self.stds),
                              np.array(self.means)+np.array(self.stds),color='violet',alpha=0.2)
        self.ax[0].set_xlim(xlim)
        self.ax[0].set_xticks(xticks)
        self.ax[0].set_title('Mean Reward Curve')
        self.ax[0].set_xlabel('entire_nums')
        self.ax[0].set_ylabel('Rewards')
        self.ax[0].legend()

        self.ax[1].clear()
        self.ax[1].plot(time, self.total_episode_rewards, color='blue',label="total")
        self.ax[1].plot(time, self.max_reward, color='red',label="Max")
        self.ax[1].plot(time, self.last_reward, color='deeppink',label="Last")
        self.ax[1].set_xlim(xlim)
        self.ax[1].set_xticks(xticks)
        self.ax[1].set_title('Reward Curve')
        self.ax[1].set_xlabel('entire_nums')
        self.ax[1].set_ylabel('Rewards')
        self.ax[1].legend()
        plt.subplots_adjust(hspace=0.5)
        plt.pause(0.001)

        plt.savefig(self.path + "{}_PoisonRec-nonorm.png".format(self.recommenderName), dpi=300)
        # return super()._on_rollout_end()
    

class MyEnv(gym.Env):
    def __init__(self, item_num, fakeUser, maliciousFeedbackNum, recommender, targetItem):
        super(MyEnv, self).__init__()

        self.item_num = item_num
        self.fakeUserNum = len(fakeUser)
        self.recommender = recommender
        self.targetItem = targetItem
        self.maliciousFeedbackNum = maliciousFeedbackNum
        self.fakeUser = fakeUser
        self.item_embeddings = recommender.model.embedding_dict['item_emb']
        self.emb_size = int(recommender.args.emb_size)

        self.observation_space = spaces.Dict({"userId":spaces.Discrete(self.fakeUserNum), "itemInteract":spaces.MultiBinary((self.fakeUserNum, item_num))})
        # self.observation_space = spaces.MultiBinary(item_num)
        # self.action_space = spaces.Discrete(item_num)
        self.action_space = spaces.Discrete(item_num)

        # self.state_dim = self.observation_space.shape  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
    
        self.if_discrete = True
        self.itemList = self.targetItem
        # self.state = [0, np.zeros(self.item_num)]
        # self.state[1][self.itemList] = 1
        self.state = {"userId":0, "itemInteract":np.zeros((self.fakeUserNum, self.item_num))}
        self.state["itemInteract"][:,self.targetItem] = 1
        self.fakeUserDone = False
        self.one_fakeUserDone = False
        self.fakeUserid = 0
        self.train = True
        self.time = 0
        self.re = []
        self.all_epi = 0

    def mode(self):
        self.train = not self.train

    def reset(self):
        self.re = []
        self.reward = 0
        self.time = 0  # 记录每一个用户产生了多少个动作
        self.one_fakeUserDone = False
        if self.fakeUserDone:
            self.fakeUserDone = False
            self.fakeUserid = 0
            if self.train:
                self.recommender.data.reset()
            self.state = {"userId":0, "itemInteract":np.zeros((self.fakeUserNum, self.item_num))}
            self.state["itemInteract"][:,self.targetItem] = 1
        return self.state 

    #表示物品在矩阵中的序号
    def step(self, action):
        self.time += 1
        # ones_indices = np.where(action == 1)[0]
        # num = self.maliciousFeedbackNum-len(self.targetItem)
        # if len(ones_indices) > num:
        #     keep_indices = np.random.choice(ones_indices, size=num, replace=False)
        #     action = np.zeros_like(action)
        #     action[keep_indices] = 1
        self.state["itemInteract"][self.state["userId"],action] = 1
        # self.itemList = np.where(self.state["itemInteract"][self.state["userId"]] == 1)[0]
        # self.fakeUserInjectChange(self.recommender, self.fakeUserid)
        self.fakeUserInjectChange(self.recommender, action)
        if self.time == self.maliciousFeedbackNum:
            optimizer = torch.optim.Adam(self.recommender.model.parameters(), lr= self.recommender.args.lRate)
            self.recommender.train(Epoch=20, optimizer=optimizer, evalNum=4)
            attackmetrics = AttackMetric(self.recommender, self.targetItem, [50])
            reward = attackmetrics.hitRate()[0]
            self.one_fakeUserDone = True
            self.re.append(reward)
            print(f"当前fakeUserid:{self.fakeUserid}   当前动作序号：{self.time}   产生的动作:{action}   奖励为:{reward}")
            self.fakeUserid = (self.fakeUserid + 1) % self.fakeUserNum
            self.state["userId"] = self.fakeUserid

            if self.fakeUserid == self.fakeUserNum - 1:  #每个虚假用户都产生了动作
                self.fakeUserDone = True
                self.all_epi += 1
                if self.all_epi % 10 == 0 and self.fakeUserDone:
                    path = ('./data/poison/' + self.recommender.dataName + "/{}/" + "episode_" + str(self.all_epi) + "_").format(self.recommender.model_name)
                    dataSave(self.recommender.data.matrix(), path + "fakedata_max{}_mean{}_total{}_last{}.txt".format(max(self.re), np.mean(self.re), sum(self.re), reward)
                            , self.recommender.data.id2user, self.recommender.data.id2item)
        else:
            reward = 0
            print(f"当前fakeUserid:{self.fakeUserid}   当前动作序号：{self.time}   产生的动作:{action}   奖励为:{reward}")
        
        info = {"end_total":self.fakeUserDone}
        return self.state, reward, self.one_fakeUserDone, info

    def fakeUserInjectChange(self, recommender, action):
        self.userNum = recommender.data.user_num
        self.itemNum = recommender.data.item_num
        # uiAdj = recommender.data.interaction_mat
        # uiAdj2 = uiAdj[:, :]
        # uiAdj2[self.fakeUser[fakeUserId],:] = 0
        # uiAdj2[self.fakeUser[fakeUserId],self.itemList] = 1
        # ui_adj = sp.csr_matrix(([], ([], [])), shape=(
        #     self.userNum + self.itemNum, self.userNum  + self.itemNum),
        #                         dtype=np.float32)
        # ui_adj[:self.userNum , self.userNum:] = uiAdj2
        recommender.data.invert("fakeuser_{}".format(self.state["userId"]), action)
        # recommender.model._init_uiAdj(ui_adj + ui_adj.T)

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.user_embedding = nn.Embedding(observation_space.spaces["userId"].n, features_dim)
        self.item_embedding = nn.EmbeddingBag(observation_space.spaces["itemInteract"].n[1], features_dim, padding_idx=0)
        
    def forward(self, observations):
        # offsets = torch.tensor([0])
        # user = observations["userId"].long()
        # E_u = self.user_embedding(user)
        # items = observations["itemInteract"][:user+1].long().squeeze()
        # items = torch.where(items==1)[0]
        # if E_u.dim() == 2:
        #     E_u = E_u.unsqueeze(0)
        # E_i = E_i.unsqueeze(1)
        # return torch.vstack((E_u, E_i)), self.item_embedding.weight
        user = observations["userId"].long()
        E_u = self.user_embedding(user)
        items = observations["itemInteract"][0][:user+1].long()
        rows, cols = torch.where(items == 1)
        new_items = items.clone()
        new_items[rows, cols] = cols
        E_i = self.item_embedding(new_items)
        if E_u.dim() == 2:
            E_u = E_u.unsqueeze(0)
        E_i = E_i.unsqueeze(1)
        return torch.vstack((E_u, E_i)), self.item_embedding.weight

class CustomPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.observation_space = args[0]
        self.action_space = args[1]
        self.targetItem = kwargs["targetItem"]
        super(CustomPolicy, self).__init__(*args, **kwargs)
        

    def forward(self,obs ,deterministic=False):
        features = self.extract_features(obs)
        E_st, E_item = self.lstm_extractor(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # E_st, E_item = self.extract_features(obs)
        # distribution = self._get_action_dist_from_latent(E_st, E_item)
        # # E_st, E_item = self.lstm_extractor(obs)
        # if self.share_features_extractor:
        #     lstm_h = self.action_net.get_lstm_h()
        #     _, latent_vf = self.mlp_extractor(lstm_h)
        # else:
        #     pi_features, vf_features = lstm_h, lstm_h
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        
        # probs = self.action_net(E_st, E_item)
        # dist = Multinomial(self.total_actions,probs=probs)
        # actions = dist.sample()
        distribution = self._get_action_dist_from_latent(E_st, E_item)
        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        while actions in self.targetItem:
            actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.action_dist = DistributionEx(self.action_space.n)

        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        #LSMNet
        self.action_net = self.action_dist.proba_distribution_lstmnet(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.lstm_extractor = CustomFeaturesExtractor(self.observation_space)
    def _get_action_dist_from_latent(self, E_st, E_item) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(E_st, E_item)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        values = torch.empty(size=(0,)).to(self.device)
        log_probs = torch.empty(size=(0,)).to(self.device)
        entropys = torch.empty(size=(0,)).to(self.device)
        for i in range(len(obs["userId"])):
            temp = {a:obs[a][i] for a in obs.keys()}
            temp["itemInteract"].unsqueeze_(0)

            features = self.extract_features(temp)
            E_st, E_item = self.lstm_extractor(temp)
            # E_st, E_item = self.extract_features(temp)
            distribution = self._get_action_dist_from_latent(E_st, E_item)
            if self.share_features_extractor:
                # lstm_h = self.action_net.get_lstm_h()
                _, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features, features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            log_prob = distribution.log_prob(actions[i])
            value = self.value_net(latent_vf)
            entropy = distribution.entropy()
            values = torch.cat((values, value))
            log_probs = torch.cat((log_probs, log_prob))
            entropys = torch.cat((entropys, entropy))
        return values, log_probs, entropys
        # features = self.extract_features(obs)
        # E_st, E_item = self.lstm_extractor(obs)
        # if self.share_features_extractor:
        #     _, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # distribution = self._get_action_dist_from_latent(E_st, E_item)
        # log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        # entropy = distribution.entropy()
        # return values, log_prob, entropy
    
    def get_distribution(self, obs):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        E_st, E_item = self.lstm_extractor(obs)
        features = super().extract_features(obs, self.pi_features_extractor)
        return self._get_action_dist_from_latent(E_st, E_item)
    
    # def predict_values(self, obs: PyTorchObs) -> th.Tensor:
    #     """
    #     Get the estimated values according to the current policy given the observations.

    #     :param obs: Observation
    #     :return: the estimated values.
    #     """
    #     E_st, E_item = super().extract_features(obs, self.vf_features_extractor)
    #     distribution = self._get_action_dist_from_latent(E_st, E_item)
    #     lstm_h = self.action_net.get_lstm_h()
    #     latent_vf = self.mlp_extractor.forward_critic(lstm_h)
    #     return self.value_net(latent_vf)

class DistributionEx(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dims: int):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution_lstmnet(self, latent_dim) -> nn.Module:
        return LSTMNet(latent_dim)

    def proba_distribution(self, action_logits: th.Tensor):
        self.distribution = Categorical(action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # return self.distribution.log_prob(actions).sum(dim=1)
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        # return self.distribution.entropy().sum(dim=1)
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.round(self.distribution.probs)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor):
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

class LSTMNet(nn.Module):
    def __init__(self,feature_dim = 64):
        super(LSTMNet, self).__init__()
        # Policy network
        self.dim = feature_dim
        self.DNN = nn.Sequential(nn.Linear(self.dim, self.dim),
        nn.ReLU(), nn.Linear(self.dim, self.dim), nn.ReLU())
        self.LSTM = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=2)

    def forward(self, E_st, E_item):
        h_t, _ = self.LSTM(E_st)
        self.last = h_t[-1,:,:]
        if self.last.dim()==2:
            self.last = self.last.unsqueeze(0)
        output = torch.softmax(self.DNN(self.last) @ E_item.T, dim=1)
        return output
    
    def get_lstm_h(self):
        return self.last
