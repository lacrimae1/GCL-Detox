import os, emoji, torch, warnings, time, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.metrics import accuracy_score
from util.DataLoader import DataLoader
from util.tool import dataSave, Picture
from util.logger import Log
from copy import deepcopy
# torch.serialization.add_safe_globals
warnings.filterwarnings("ignore")

class Model():
    def __init__(self, recommendModel, recommendArg, attackModel=None, attackArg=None, method=[], T1=None, is_visual=True,three=False,T2=None):

         # Evaluation metrics
        self.hitRate = []
        self.precision=[]
        self.recall = []
        self.ndcg = []
        self.RecommendTestResult = []

        # measure:defaultdict(dict) measure[topk][indicator] = num
        # self.measure_result.append(measure)
        self.measure_result = {} # 0:NoneAttack, i:attackSize=i
        self.HR = {}
        self.uniformity = {}
        self.best_epoch = {}
 
        # Record the process
        self.result = list()

        # Recommend model setup
        self.recommendModel = recommendModel
        self.recommendModelName = recommendArg.model_name
        self.datasetName = recommendArg.dataset
        self.recommendArg = recommendArg
        top = self.recommendArg.topK.split(',')
        self.topN = [int(num) for num in top]

        self.gamma_2 = {'ml-1M':1.25,'gowalla':0.02,'douban':0.025}
        self.gamma_3 = 0.3
        self.stage_args = {'cut':False,'gamma_2':0.25,'gamma_3':self.gamma_3}

        # Attack model setup
        self.attackModel = attackModel
        self.attackModelName = attackArg.attackModelName
        self.maliciousUserSize = list(map(lambda x: int(x)/100, attackArg.maliciousUserSize.split(",")))
        self.maliciousFeedbackSize = attackArg.maliciousFeedbackSize
        self.times = attackArg.times
        self.poisonDatasetOutPath = attackArg.poisonDatasetOutPath

        # Target attack setup
        self.attackTargetChooseWay = attackArg.attackTargetChooseWay
        self.targetSize = attackArg.targetSize

        # Gradient attack needs adjacency matrix gradient
        self.requires_grad = self.attackModel.recommenderGradientRequired

        # Determine target items
        self.targetItem = attackModel.targetItem

        path = f"./data/clean/{self.datasetName}/sample.pickle"
        if not os.path.exists(path) and self.recommendModelName=="LightGCN":
            sample = {}
            self.hot_user, self.cold_user, self.hot_item, self.cold_item = self.recommendModel.data.randomly_sample()
            self.cold_item = np.unique(np.hstack([self.cold_item, np.array(self.targetItem)]))
            self.sampled_user = self.recommendModel.data.randomly_sample(sample_type=2)
            sample["hot_user"] = self.hot_user
            sample["cold_user"] = self.cold_user
            sample["hot_item"] = self.hot_item
            sample["cold_item"] = self.cold_item
            sample["sampled_user"] = self.sampled_user
            with open(path,"wb") as file:
                pickle.dump(sample, file)
        else:
            with open(path,"rb") as file:
                sample = pickle.load(file)
            self.hot_user = sample["hot_user"]
            self.cold_user = sample["cold_user"] 
            self.hot_item = sample["hot_item"]
            self.cold_item = sample["cold_item"]
            self.sampled_user = sample["sampled_user"]
        self.method = "_" + "_".join(method) if len(method)!=0 else ""
        self.Met = method
        self.T1 = T1
        self.T2 = T2
        self.is_visual = is_visual
        self.method += "_three" if three else ""
        if self.T1 is not None: 
            self.method += '_' + str(self.T1)
        else:
            self.method += '_origin'
            
        self.method += f"_T2_{T2}" if T2 else ""
        # Logger file
        head = self.recommendModelName + self.method + " attack by " + self.attackModelName
        file = f'./log/{self.attackModelName}/{self.datasetName}/{self.datasetName+self.method}/'
        self.logFilename = self.recommendModelName + self.method + "_" + self.attackModelName + "_" + self.datasetName
        self.logger, self.formatter = Log(head=head, filename=self.logFilename, file=file).get()

        # Basic information in logger file
        message = "\n" + "-" * 10 + "Recommend Model Infomation" + "-" * 10 + "\n"
        for i in recommendArg._get_kwargs():
            message += str(i[0]) + ":" + str(i[1]) + "\n"
        if self.T1 is not None: message += f"先训练infoNCE: {self.T1}次\n"
        self.logger.info(message)
        # print(message)
        
        self.formatter.use_format = False
        if attackArg is not None:
            message = "-" * 10 + "Attack Model Infomation" + "-" * 10 + "\n"
            for i in attackArg._get_kwargs():
                message += str(i[0]) + ":" + str(i[1]) + "\n"
            message += "TargetItemSeq:{}".format(attackModel.targetItem) + "\n"
            message += "FakeUserNum:{}".format(attackModel.fakeUserNums) + "\n"
            message += "MaliciousFeedbackNum:{}".format(attackModel.maliciousFeedbackNum) + "\n"
            self.logger.info(message)
            # print(message)

    def update(self, recommendModel, saved):
        # recommendModel.measure = saved["measure"]
        # recommendModel.stds_all = saved["stds_all"]
        # recommendModel.stds_k = saved["stds_k"]
        # recommendModel.targetHR = saved["targetHR"]
        recommendModel.best_epoch = saved["best_epoch"]
        recommendModel.user_emb = saved["user_emb"]
        recommendModel.item_emb = saved["item_emb"]
        # recommendModel.model.embedding_dict = saved["emb_dict"]
        # for i in saved:
        #     print(f"saved[{i}]=")
        return recommendModel
    
    def size(self, path):
        import math
        size = os.path.getsize(path)
        units = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size, 1024)))
        size = round(size / (1024 ** i), 2)
        print(f"{size}{units[i]}")

    def RecommendTrain(self, attack=None, logger = None):
        # symbols = ["::zero::","::one::","::two::","::three::","::four::","::five::"]
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{'-'*10} Now time is {t} {'-'*10}")
        sim_path = f"./simple_modelsaved/{self.attackModelName}/{self.datasetName}/{self.datasetName+self.method}/{self.recommendModelName}/"
        if not os.path.exists(sim_path):
            os.makedirs(sim_path)
        if attack is None:
            self.formatter.use_format = True
            path = f"./modelsaved/{self.attackModelName}/{self.datasetName}/" + self.datasetName + self.method + "/" + self.recommendModelName + "/"
            self.logger.info(emoji.emojize(f"\n{'-'*10}:revolving_hearts:Recommending on Clean Dataset:revolving_hearts:{'-'*10}\n"))
            self.formatter.use_format = False
            if not os.path.exists(path):
                os.makedirs(path)
            simple_path = sim_path + f"clean_{self.recommendModelName}"+self.method
            if not os.path.exists(simple_path):
                print(f"model_path:{simple_path}")
                self.logger.info(f"Model_path:{simple_path}")
                logger.info(f"Model_path:{simple_path}")
                self.logger.info("Model is not build before, building and training...")
                logger.info("Model is not build before, building and training...")
                picture=Picture(self.recommendModelName,self.datasetName,self.method,self.attackModelName)
                evalNum = 2 if len(self.Met)!=0 else 5
                if self.T1 is not None: self.recommendModel.T1 = self.T1
                if self.T2 is not None: self.recommendModel.T2 = self.T2
                saved = self.recommendModel.train(picture=picture, evalNum=2)
                torch.save(saved, simple_path)
                recommendModel = self.recommendModel
                self.logger.info(f"Model is saved. Path:{simple_path}")
                logger.info(f"Model is saved. Path:{simple_path}")
                if self.is_visual:self.visualize(recommendModel.user_emb, recommendModel.item_emb)
            else:
                print("Model is exist in {}, loading...".format(simple_path))
                self.logger.info("Model is exist in {}, loading and training...".format(simple_path))
                logger.info("Model is exist in {}, loading and training...".format(simple_path))
                print("Model has already loaded.")
                self.size(simple_path)
                saved = torch.load(simple_path)
                self.recommendModel = self.update(self.recommendModel, saved)
                recommendModel = self.recommendModel
                if self.is_visual:self.visualize(recommendModel.user_emb, recommendModel.item_emb)
            logger.info(emoji.emojize(f":dizzy:./result_show/{self.attackModelName}/distribution/{self.datasetName+ self.method}/{self.recommendModelName}.png is finishing..."))
            logger.info(emoji.emojize(":love_letter: clean_{} endding...\n".format(self.recommendModelName+self.method)))
            self.logger.info(f"The Best Epoch is {recommendModel.best_epoch}")
            self.logger.info(emoji.emojize(':hundred_points:Test Performance:hundred_points:'))
            # gpu = GPUtil.getGPUs()[0]
            # print(f"显存占用情况：GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            # gini = self.cal_popularity_bias(recommendModel.data, recommendModel.user_emb, recommendModel.item_emb)
        else:
            self.formatter.use_format = True
            # Recommend in poisoning data
            poisonArg = deepcopy(self.recommendArg)
            # poisonArg.dataset = self.poisonDataName + "/" + self.recommendModelName + "/" + self.attackModelName + "/" + str(attack) + ".txt"
            poisonArg.data_path = f"data/poison/"
            datapath = f'./data/poison/' + self.poisonDataName + f"attacksize_{attack}_train.txt"
            print(f"poisonDatapath={datapath}")
            self.logger.info(emoji.emojize(f"\n{'-'*10} :mending_heart:Recommending on Poisoning Dataset:mending_heart: (AttackSize:{attack}%) {'-'*10}"))
            self.formatter.use_format = False
            self.logger.info(f"Attack Training Data Path:{datapath}")
            simple_path = f"./simple_modelsaved/{self.attackModelName}/{self.datasetName}/{self.datasetName+self.method}/{self.recommendModelName}/attackSize_{attack}_{self.attackModelName}"+self.method
            path_without_train = f"./data/clean/{self.datasetName}/"
            poisonData = DataLoader(poisonArg, self.Met, path=datapath, pathvt=path_without_train)
            if not os.path.exists(simple_path):
                print(f"model_path:{simple_path}")
                logger.info("Model is not build before, building and training...")
                self.logger.info("Model is not build before, building and training...")
                self.recommendModel.__init__(poisonArg, poisonData, self.attackModelName, self.Met)
                if self.T1 is not None: self.recommendModel.T1 = self.T1
                if self.T2 is not None: self.recommendModel.T2 = self.T2
                recommendModel = self.recommendModel
                picture = Picture(self.recommendModelName, self.datasetName, self.method, self.attackModelName, str(attack))
                evalNum = 2 if len(self.Met)!=0 else 5
                if self.requires_grad:
                    self.grad = recommendModel.train(requires_grad=self.requires_grad, picture=picture, attack=attack)
                else:
                    saved = recommendModel.train(picture = picture, evalNum=2, attack=attack)
                torch.save(saved, simple_path)
                logger.info(f"Model is saved. Path:{simple_path}")
                logger.info(f"The Best Epoch is {recommendModel.best_epoch}")
                self.logger.info(f"Model is saved. Path:{simple_path}")
                if self.is_visual:self.visualize(recommendModel.user_emb, recommendModel.item_emb, attack="_"+str(attack))
            else:
                print("Model is exist in {}, loading...".format(simple_path))
                self.logger.info(f"Model is exist in {simple_path}, loading and training...")
                logger.info(f"Model is exist in {simple_path}, loading and training...")
                print("Model has already loaded.")
                self.size(simple_path)
                saved = torch.load(simple_path)
                recommendModel = self.update(self.recommendModel, saved)
                logger.info(f"The Best Epoch is {recommendModel.best_epoch}")
                self.logger.info(f"The Best Epoch is {recommendModel.best_epoch}")
                if self.is_visual:self.visualize(recommendModel.user_emb, recommendModel.item_emb, attack="_"+str(attack))
            logger.info(emoji.emojize("poison_{} endding...{}%\n".format(self.recommendModelName+self.method, attack)))
            self.logger.info(emoji.emojize(':hundred_points: Test Performance :hundred_points:'))
        # sim = self.cal_uniformity(torch.cat([recommendModel.user_emb, recommendModel.item_emb],0))
        # gini = self.cal_popularity_bias(poisonData, recommendModel.user_emb, recommendModel.item_emb)
        # self.uniformity.append(sim)
        is_pop = False
        measure, stds_all, stds_k, hrs = recommendModel.test(recommendModel.data.test_set, flag=True, is_pop=is_pop)
        y = 0 if attack is None else attack
        self.measure_result[y] = measure
        self.HR[y] = hrs
        self.best_epoch[y] = recommendModel.best_epoch
        self.logger.info(f"The best epoch is {recommendModel.best_epoch}")
        print(f"model_path:{simple_path}")
        # if attack is None: self.frequency = recommendModel.frequency
        # self.po_bias.append(gini)
        for t, i in enumerate(measure):
            self.logger.info(f"Top{i}:{measure[i]} " + f"targetHR:{hrs[t]}")
        del recommendModel
        torch.cuda.empty_cache()

    def PoisonDataAttack(self, logger):
        self.logger.info(f"Poison Data Generating...{emoji.emojize(':face_with_monocle:')}")
        logger.info(f"Poison Data Generating...{emoji.emojize(':face_with_monocle:')}")
        print("_"*80)
        print("Poison Data Generating...")
        # AttackSize/train.txt
        self.poisonDataName = self.datasetName + "/" + self.attackModelName + "/"+ self.recommendModelName + "/" 
        if not os.path.exists(f'./data/poison/' + self.poisonDataName):
            os.makedirs(f'./data/poison/' + self.poisonDataName)
        self.logger.info(emoji.emojize(f"{'-'*10}:ghost:Poisoning Data generating...:ghost:{'-'*10}"))
        logger.info(emoji.emojize(f"{'-'*10}:ghost:Poisoning Data generating...:ghost:{'-'*10}"))
        self.formatter.use_format = False
        # logger1, formatter1 = Log(head=f"{self.attackModelName} vs {self.recommendModelName}", filename=f"attackModelInfo_{self.recommendModelName}", file="./log/", mode='a').get()
        for i in range(5):
            self.logger.info(emoji.emojize("---:sos: Current attackSize:" + str(i+1) + "% :sos:---"))
            logger.info(emoji.emojize("---:sos: Current attackSize:" + str(i+1) + "% :sos:---"))
            path = f'./data/poison/' + self.poisonDataName + f"attacksize_{i+1}_"
            # formatter1.use_format = True
            # logger1.info(emoji.emojize("---:sos: Current attackSize:" + str(i+1) + "% :sos:---"))
            # formatter1.use_format = False
            if not os.path.exists(path+"train.txt"):
                if self.requires_grad:
                    poisonRatings = self.attackModel.posionDataAttack(self.grad, attack=i)
                elif self.attackModel.recommenderModelRequired:
                    poisonRatings = self.attackModel.posionDataAttack(deepcopy(self.recommendModel), attack=i)
                else:
                    poisonRatings = self.attackModel.posionDataAttack(attack=i)
                # save train.txt,test.txt,val.txt in posion data path
                
                dataSave(poisonRatings, path + "train.txt", self.recommendModel.data.id2user, self.recommendModel.data.id2item)
            else:
                # logger1.info("Poisoning Data is exist in {}".format(path+"train.txt"))
                self.logger.info("Poisoning Data is exist in {}".format(path+"train.txt"))
                logger.info("Poisoning Data is exist in {}".format(path+"train.txt"))
            self.logger.info(f"Data attack Size_{i+1}% by {self.attackModelName} has done.")
            logger.info(f"Data attack Size_{i+1}% by {self.attackModelName} has done.")
            print(f"Data attack Size_{i+1}% by {self.attackModelName} has done.")
            torch.cuda.empty_cache()
        logger.info("")
        self.logger.info("")
        
    def visualize(self, user_emb, item_emb, attack=""):
        sns.set(style='whitegrid', font_scale=1.1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7),
                               gridspec_kw={'height_ratios':[2,1]})

        recommenderName = self.recommendModelName
        targetItem = self.attackModel.targetItem
        path = f"./result_show/{self.attackModelName}/{self.datasetName}/distribution/{self.datasetName+ self.method}/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        hot_user_emb, cold_user_emb = user_emb[self.hot_user], user_emb[self.cold_user]
        hot_item_emb, cold_item_emb = item_emb[self.hot_item], item_emb[self.cold_item]
        embedding = torch.cat((hot_user_emb, cold_user_emb, hot_item_emb, cold_item_emb),0)
        embedding = embedding.detach().cpu().numpy()
        tsne = manifold.TSNE(perplexity=100, n_components=2, init='pca',random_state=2025)
        X_tsne = tsne.fit_transform(embedding)

        hot_user_label, cold_user_label = np.array([0]*len(self.hot_user)), np.array([1]*len(self.cold_user))
        hot_item_label, cold_item_label = np.array([2]*len(self.hot_item)), np.array([3]*len(self.cold_item))
        labels = np.concatenate((hot_user_label, cold_user_label, hot_item_label, cold_item_label))
        palette = {0:'#FF4B4B', 1:'#4B7BFF', 2:'#FFA600', 3:'#00B3E6',4:'black'}
        angles = np.arctan2(X_tsne[:,1], X_tsne[:,0])   # [-pi, pi]
        angle_df = {'Angle': angles, 'Type': labels}
        sns.kdeplot(data=angle_df, x='Angle', hue='Type',
                    palette=palette, fill=True, alpha=.3,
                    bw_adjust=.8, ax=ax2)
        ax2.set_xlabel('Angle')
        ax2.set_ylabel('Density')
        ax2.set_xlim(-np.pi, np.pi)
        ax2.set_title('Angle Distribution KDE')
        ax2.legend(loc='upper right', fontsize=9)

        targetIndex = [np.where(self.cold_item == x)[0][0] for x in targetItem] #返回cold_item中是目标物品的索引
        cold_item_label[targetIndex] = 4
        labels = np.concatenate((hot_user_label, cold_user_label, hot_item_label, cold_item_label))
        ax1.scatter(X_tsne[labels == 0, 0],X_tsne[labels==0, 1],s=10,c='#FF4B4B',label='Hot Users',alpha=0.5)
        ax1.scatter(X_tsne[labels == 1, 0],X_tsne[labels==1, 1],s=10,c='#4B7BFF',label='Cold Users',alpha=0.5)
        ax1.scatter(X_tsne[labels == 2, 0],X_tsne[labels==2, 1],s=10,c='#FFA600',label='Hot Items',alpha=0.5)
        ax1.scatter(X_tsne[labels == 3, 0],X_tsne[labels==3, 1],s=10,c='#00B3E6',label='Cold Items',alpha=0.5)
        ax1.scatter(X_tsne[labels == 4, 0],X_tsne[labels==4, 1],s=10,c='black',label='targets',alpha=1)
        ax1.set_title(f"{recommenderName}")
        ax1.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(path+f"{recommenderName+attack}.png", dpi=200)
    
    def cal_uniformity(self, emb, t=2):
        # with torch.no_grad():
        #     embedding = torch.cat([user_emb, item_emb],0).detach().cpu()
        # res = 0.0
        # for s in range(0, embedding.shape[0],1000):
        #     emb = embedding[s:s+1000] if s+1000<embedding.shape[0] else embedding[s:embedding.shape[0]]
        #     a = torch.exp(-2*torch.pow(torch.norm(emb.unsqueeze(1)-emb.unsqueeze(0),dim=2),2))
        #     a -= torch.eye(emb.shape[0])
        #     b = torch.sum(a) / 2
        #     res += b
        # res = torch.log(torch.tensor(res)).numpy()
        def chunked_pdist(emb, chunk=2048):
            n = emb.shape[0]
            out = []
            for i in range(0, n, chunk):
                a = emb[i:i+chunk]
                out.append(torch.cdist(a, emb, p=2).pow(2).sum(1))
            return torch.cat(out)
        emb = F.normalize(emb, p=2, dim=-1)
        emb = emb[self.sampled_user]
        try:
            sq_dist = torch.pdist(emb, p=2)
        except:
            sq_dist = chunked_pdist(emb)
        res = sq_dist.mul(-t).exp().mean().log()
        return res.detach().cpu().item()

    def cal_popularity_bias(self, data, user_emb, item_emb):
        Train = data.create_sparse_matrix(data=data.val_data+data.training_data)
        Test = data.create_sparse_matrix(data=data.test_data)
        recall_denom = np.array([max(np.sum(Test[:,j]),1) for j in range(len(data.item))])
        pops = np.sum(Train,axis=0)
        item_sorted = np.argsort(pops)
        ratings = user_emb @ item_emb.T
        M = data.item_num
        for i, user in enumerate(data.test_set):
            rated = data.user_rated_id(user)
            ratings[data.user[user]][rated] = -10e8
        gini=dict()
        for n in self.topN:
            _, indices = torch.topk(ratings, n, dim=1)
            mask = torch.zeros_like(ratings, dtype=torch.bool)
            batch_indices = torch.arange(ratings.shape[0]).unsqueeze(1).expand(-1,n)
            mask[batch_indices, indices] = True
            pred = mask.int()
            true_positives = torch.sum(pred*torch.tensor(Test.toarray()).cuda(),axis=0).detach().cpu().numpy()
            recall_denom = np.array([max(np.sum(Test[:,j]),1) for j in range(data.item_num)])
            recalls = true_positives / recall_denom
            items_ordered_by_pop = np.squeeze(recalls[item_sorted])
            coefs = np.array([2*i-M-1 for i in range(1,M+1)])
            gc = np.dot(coefs, items_ordered_by_pop) /(M * np.sum(items_ordered_by_pop))
            gini[n] = gc
        return gini

    def SVM(self, user_emb, item_emb):
        X_train = self.recommendModel.data.X_train
        X_test = self.recommendModel.data.X_test
        y_train = np.array(self.recommendModel.data.y_train)
        y_test = np.array(self.recommendModel.data.y_test)
        with torch.no_grad():
            X1 = torch.cat((user_emb[torch.tensor(X_train[0])],item_emb[torch.tensor(X_train[1])]),dim=1).detach().cpu().numpy()
            X2 = torch.cat((user_emb[torch.tensor(X_test[0])],item_emb[torch.tensor(X_test[1])]),dim=1).detach().cpu().numpy()
        SVM = SVC(kernel='rbf',C=1)
        SVM.fit(X1, y_train)
        y_pred = SVM.predict(X2)
        acc = accuracy_score(y_test, y_pred)
        print(f"acc={acc}")