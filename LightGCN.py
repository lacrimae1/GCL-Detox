from conf.recommend_parser import recommend_parse_args
from conf.attack_parser import attack_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
from util.logger import Log
from framework import Model
from create_table import Table
import matplotlib.pyplot as plt
import matplotlib, os, emoji, logging, pickle
import sys

def pattern(new=True, method=[], W = None, Method=""):
    if new:
        recommend_args.lRate = lr[name]
        recommend_args.maxEpoch = epoch
    
    recommend_args.model_name = name
    recommend_args.dataset = datasetName
    attack_args.attackModelName = attackModelName

    import_str = 'from recommender.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)
    import_str = 'from attack.' + attack_args.attackCategory + "." + attack_args.attackModelName + ' import ' + attack_args.attackModelName
    exec(import_str)

    data = DataLoader(recommend_args, method)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data, attackModelName, method)
    attack_model = eval(attack_args.attackModelName)(attack_args, data)
    recommend_model.add(attack_model.targetItem_name, attack_model.targetItem)
    framework = Model(recommend_model, recommend_args, attack_model, attack_args, method, W=W, is_visual=is_visual)

    formatter.use_format = True
    logger.info(emoji.emojize('\n' + '-'*10 + ':face_with_monocle:' + name + Method + " recommend infomation...:face_with_monocle:" + '-'*10))
    formatter.use_format = False

    logger.info(emoji.emojize(":palm_tree:" + name + Method + " recommend on clean data..."))
    framework.RecommendTrain(logger=logger)
    framework.PoisonDataAttack(logger=logger)
    for size in range(framework.times):
        logger.info(emoji.emojize(":fallen_leaf:" + name + Method + f" recommend on poisoning data, attackSize:{size+1}%..."))
        framework.RecommendTrain(attack=size+1,logger=logger)
    
    measures[name+Method] = framework.measure_result
    # stds[name] = framework.stds_all
    # stds_k[name] = framework.stds_topk
    targetHR[name+Method] = framework.HR
    best_epoch[name+Method] = framework.best_epoch
    uniformity[name+Method] = framework.uniformity
    # user_uniformity[name+Method] = framework.user_uniformity
    # item_uniformity[name+Method] = framework.item_uniformity
    # popu_bias[name+Method] = framework.po_bias
    framework.logger.setLevel(logging.CRITICAL)
    return framework

cols = [["0","1%","2%","3%","4%","5%"]]
rows = [["ml-1M","douban"],["LightGCN", "SimGCL", "XSimGCL", "SSL4Rec", "SGL"]]
rNum = [len(ele) for ele in rows]
if __name__ == '__main__':
    arguments = sys.argv[1:]
    # Logger files
    method = ["new",'50']
    Method = "_" + "_".join(method)
    temp = []
    for ind in ["NDCG", "HR"]:
        temp.append(ind)
        temp.append(ind + Method)
    cols.append(temp)
    cNum = [len(ele) for ele in cols]
    Data = ["ml-1M", "douban", "gowalla","Yelp","Epinions"]
    Attack = ["CLeaR", "PGA","A_ra","RandomAttack"]
    # for a in range(3):
    #     for b in range(4):
    a=4
    b=0
    DatasetName = [Data[a]]
    attackName = [Attack[b]]
    # "LightGCN", "SSL4Rec", "SimGCL", "XSimGCL",
    recommenderName = ["LightGCN"]
    attackType = {"RandomAttack": "Black", "CLeaR": "White", "PGA": "White", "A_ra": "Gray","PipAttack":"White"}
    color = ['gray', 'green', 'blue','pink','orange','cyan']
    n = 0.002
    lr ={"SimGCL":n, "XSimGCL":n, "SGL":0.001,"SSL4Rec":n}
    epoch = 300
    is_visual = True
    # 记录数据,生成表格

    for attackModelName in Attack:
        for datasetName in Data[:-2]:
            
            measures = dict() # key:name, value:framework.measure_result([measure[topk][indicator]]*5)
            stds = dict() # key:name, value:framework.stds([stds]*len(test_data))
            targetHR = dict() #key:name value:framework.HR([noneattack[top10hr,top20hr,top30hr,top50hr],1_attack[]])
            stds_k = dict()
            uniformity = dict()
            # user_uniformity = dict()
            # item_uniformity = dict()
            best_epoch = dict()
            # popu_bias = dict()
            total = dict()

            recommend_args = recommend_parse_args()
            attack_args = attack_parse_args()
            recommend_args.gpu_id = '0'
            attack_args.attackCategory = attackType[attackModelName]
            os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
            seed = recommend_args.seed
            seedSet(seed)
            indicator_path = f"./indicator_saved_three/{attackModelName}/{datasetName}/"
            if not os.path.exists(indicator_path):
                os.makedirs(indicator_path)
            for c, name in enumerate(recommenderName):
                # logger file: 记录当前数据集下的所有执行过程
                head = f"information about dataset: {datasetName} vs {attackModelName}"
                filename = datasetName
                logger,formatter = Log(head=head,filename=filename+'_'+name, file=f"./log/{attackModelName}/",mode='w').get()
                # with open(indicator_path+name+".pickle", "rb") as file:
                #     total = pickle.load(file)
                # print(f"{attackModelName} {datasetName}: {total['uniformity']}")
                # 原始推荐
                framework = pattern(new=False)

                total["measures"] = measures
                total["targetHR"] = targetHR
                total["best_epoch"] = best_epoch
                total["uniformity"] = uniformity
                # # total["user_uniformity"] = user_uniformity
                # # total["item_uniformity"] = item_uniformity
                # # total["popu_bias"] = popu_bias
                print(uniformity)
                with open(indicator_path+name+".pickle", "wb") as file:
                    pickle.dump(total, file)

            logger.info(emoji.emojize(name+Method + " results picture have done! :maple_leaf:"))
            logger.info('\n')
