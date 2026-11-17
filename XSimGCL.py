from conf.recommend_parser import recommend_parse_args
from conf.attack_parser import attack_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
from util.logger import Log
from framework import Model
from create_table import Table
import matplotlib.pyplot as plt
import matplotlib, os, emoji, logging, pickle
matplotlib.use('Agg')

def pattern(new=True, method=[], T1 = None, Method="", three=False,T2=None):
    if new:
        recommend_args.lRate = lr[name]
        recommend_args.maxEpoch = epoch
        recommend_args.emb_size = 128
    
    recommend_args.model_name = name
    recommend_args.dataset = datasetName
    attack_args.attackModelName = attackModelName

    import_str = f'from {recommender_file}.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)
    import_str = 'from attack.' + attack_args.attackCategory + "." + attack_args.attackModelName + ' import ' + attack_args.attackModelName
    exec(import_str)

    data = DataLoader(recommend_args, method)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data, attackModelName, method)
    attack_model = eval(attack_args.attackModelName)(attack_args, data)
    recommend_model.add(attack_model.targetItem_name, attack_model.targetItem)
    framework = Model(recommend_model, recommend_args, attack_model, attack_args, method, T1=T1,T2=T2, is_visual=is_visual, three=three)

    formatter.use_format = True
    logger.info(emoji.emojize('\n' + '-'*10 + ':face_with_monocle:' + name_plus + Method + " recommend infomation...:face_with_monocle:" + '-'*10))
    formatter.use_format = False

    logger.info(emoji.emojize(":palm_tree:" + name_plus + Method + " recommend on clean data..."))
    if clean: framework.RecommendTrain(logger=logger)
    framework.PoisonDataAttack(logger=logger)
    # frequency[name_plus+Method] = framework.frequency
    for size in range(at_size):
        logger.info(emoji.emojize(":fallen_leaf:" + name_plus + Method + f" recommend on poisoning data, attackSize:{size+1}%..."))
        framework.RecommendTrain(attack=size+1,logger=logger)
    
    measures[name_plus+Method] = framework.measure_result
    # stds[name] = framework.stds_all
    # stds_k[name] = framework.stds_topk
    targetHR[name_plus+Method] = framework.HR
    best_epoch[name_plus+Method] = framework.best_epoch
    # uniformity[name_plus+Method] = framework.uniformity
    # popu_bias[name+Method] = framework.po_bias
    framework.logger.setLevel(logging.CRITICAL)
    return framework

if __name__ == '__main__':
    # Logger files
    attempt = "ib"
    method = [attempt]
    temp = []
    # ml-1M, douban, Yelp
    Data = ["ml-1M", "douban", "Yelp","ml-10M"]
    Attack = ["CLeaR", "PGA","A_ra","RandomAttack"]
    # for a in range(3):
    #     for b in range(4):
    a=2
    b=0
    DatasetName = [Data[a]]
    AttackName = [Attack[b]]
    # "LightGCN", "SSL4Rec", "SimGCL", "XSimGCL",
    recommenderName = ["XSimGCL"]
    attackType = {"RandomAttack": "Black", "CLeaR": "White", "PGA": "White", "A_ra": "Gray","PipAttack":"White"}
    color = ['gray', 'green', 'blue','pink','orange','cyan']
    n = 0.001
    lr ={"SimGCL":n, "XSimGCL":n, "SGL":0.001,"SSL4Rec":n}
    epoch = 300
    is_visual = False
    # 记录数据,生成表格

    for attackModelName in AttackName:
        for datasetName in DatasetName:
            base = 10

            measures = dict()
            targetHR = dict()
            best_epoch = dict()
            uniformity = dict()
            total = dict()

            recommend_args = recommend_parse_args()
            attack_args = attack_parse_args()
            recommend_args.gpu_id = '1'
            os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
            attack_args.attackCategory = attackType[attackModelName]
            seed = recommend_args.seed
            seedSet(seed)
            indicator_path = f"./indicator_saved_new/{attackModelName}/{datasetName}/"
            if not os.path.exists(indicator_path):
                os.makedirs(indicator_path)
            for c, name in enumerate(recommenderName):
                # logger file: 记录当前数据集下的所有执行过程
                head = f"information about dataset: {datasetName} vs {attackModelName}"
                filename = datasetName
                logger,formatter = Log(head=head,filename=filename+'_'+name, file=f"./log/{attackModelName}/",mode='w').get()

                # 原始推荐
                recommender_file="recommender_ori"
                name_plus = name
                at_size = 5
                clean = True
                framework = pattern(new=False)
                clean = False

                # 三阶段(at_size=1 t1=[10,20,30,40,50],t2=20 | t1=10,t2=[10,20,30,40,50])(t1=10,t2=20,at_size=[0,1,2,3,4,5])
                # recommender_file="recommender"
                # three = True
                # name_plus = name+"_plus"
                # attempt = "ib"
                # # 过渡轮数 T2 = 20
                # t2=2*base
                # flag = []
                # for t1 in range(base,6*base,base):
                #     method = [attempt]
                #     if t1 == 10 and t2==20:
                #         at_size = 5
                #     else:
                #         at_size = 1
                #     flag.append((t1,t2))
                #     Method = "_" + "_".join(method) + f"_three_{t1}_T2_{t2}"
                #     framework = pattern(method=method, T1=t1,T2=t2, Method=Method, three=three)
                #     # 遍历不同的过渡阶段
                #     if (base, t1) not in flag:
                #         Method = "_" + "_".join(method) + f"_three_{base}_T2_{t1}"
                #         framework = pattern(method=method, T1=base,T2=t1, Method=Method, three=three)

                # # 过渡轮数 T2 = 10
                # t2=base
                # flag=[]
                # for t1 in range(base,6*base,base):
                #     method = [attempt]
                #     if t1 == 10 and t2==10:
                #         at_size = 5
                #     else:
                #         at_size = 1
                #     flag.append((t1,t2))
                #     Method = "_" + "_".join(method) + f"_three_{t1}_T2_{t2}"
                #     framework = pattern(method=method, T1=t1,T2=t2, Method=Method, three=three)
                #     # 遍历不同的过渡阶段
                #     if (base, t1) not in flag:
                #         Method = "_" + "_".join(method) + f"_three_{base}_T2_{t1}"
                #         framework = pattern(method=method, T1=base,T2=t1, Method=Method, three=three)


                # # 预训练+BPR损失,没有过渡阶段(t1取10,at_size=1)(消融)
                # recommender_file="recommender_ori"
                # three = False
                # name_plus = name
                # attempt = "ib"
                # at_size = 1
                # for t1 in range(base,6*base,base):
                #     method = [attempt]
                #     Method = "_" + "_".join(method) + f"_{t1}"
                #     framework = pattern(method=method, T1=t1, Method=Method, three=three)

                # # 没有预训练阶段(t1=0,t2=20,at_size=1)
                # recommender_file="recommender_ori"
                # three = False
                # name_plus = name
                # attempt = "pre_not"
                # at_size = 1
                # for t2 in range(base,2*base,base):
                #     method = [attempt]
                #     Method = "_" + "_".join(method) + f"_{t2}"
                #     framework = pattern(method=method, T1=t2, Method=Method, three=three)
                
                # # 没有预训练阶段(t1=0,t2=10,at_size=1)
                # recommender_file="recommender_ori"
                # three = False
                # name_plus = name
                # attempt = "pre_not"
                # for t2 in range(base,2*base,base):
                #     method = [attempt]
                #     Method = "_" + "_".join(method) + f"_{t2}"
                #     framework = pattern(method=method, T1=t2, Method=Method, three=three)

                
                
                total["measures"] = measures
                total["targetHR"] = targetHR
                total["best_epoch"] = best_epoch
                # total["uniformity"] = uniformity
                # total["frequency"] = frequency
                print(f"{name} {attackModelName} {datasetName}")
                with open(indicator_path+name+".pickle", "wb") as file:
                    pickle.dump(total, file)
                
                for key, value in total.items():
                    for k in value.keys():
                        print(f"{key}  {k}")
            # logger.info(emoji.emojize(name+Method + " results picture have done! :maple_leaf:"))
            logger.info('\n')
