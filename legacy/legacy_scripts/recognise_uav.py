import os
import os.path as osp
import pickle
import sys
# import shutil

root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
from opengait.utils import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from loguru import logger

from opengait.modeling import models

import gait_compare as gc

recognise_cfgs = {  
    "gaitmodel":{
        "model_type": "Baseline",
        # "cfg_path": "./configs/baseline/baseline_GREW.yaml",
        "cfg_path": "./configs/gaitbase/gaitbase_da_dronegait1.yaml",
        # "cfg_path": "./configs/gaitbase/gaitbase_da_dronegait2.yaml"
        
    },
}

#通过传递模型类型 (model_type) 和配置文件路径 (cfg_path)
def loadModel(model_type, cfg_path):

    #动态获取 baselineDemo 类中名为 model_type：BaselineDemo 的属性或方法，并赋值给变量 Model
    Model = getattr(models, model_type)

    cfgs = config_loader(cfg_path)

    model = Model(cfgs, training=False)

    return model

def gait_sil(sils, embs_save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        embs_save_path (Path): Output path.
    Returns:
        feats (dict): Dictionary of features
    """

    gaitmodel = loadModel(**recognise_cfgs["gaitmodel"])

    gaitmodel.requires_grad_(False)
    #设置模型为评估模式（eval）
    gaitmodel.eval()
    #用于存储最终提取的步态特征
    
    feats = {}
    for inputs in sils:
        #获取输入数据
        ipts = gaitmodel.inputs_pretreament(inputs)
        
        id = inputs[1][0]
        if id not in feats:
            feats[id] = []
        type = inputs[2][0] 
        view = inputs[3][0]

        embs_pkl_path = "{}/{}/{}/{}".format(embs_save_path, id, type, view)
        if not os.path.exists(embs_pkl_path):
            os.makedirs(embs_pkl_path)
        embs_pkl_name = "{}/{}.pkl".format(embs_pkl_path, inputs[3][0])

        #调用模型的 forward 方法，将预处理后的步态序列 ipts 输入到模型中
        #retval：模型的预测值（通常与任务相关）。embs：模型提取的特征向量（嵌入特征）。
        retval, embs = gaitmodel.forward(ipts)

        pkl = open(embs_pkl_name, 'wb')
        pickle.dump(embs, pkl)
        feat = {}
        feat[type] = {}
        feat[type][view] = embs
        feats[id].append(feat)

    return feats    

'''mutil gpu
def gait_sil(sils, embs_save_path):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaitmodel = loadModel(**recognise_cfgs["gaitmodel"]).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[Info] Using {torch.cuda.device_count()} GPUs.")
        gaitmodel = torch.nn.DataParallel(gaitmodel)

    gaitmodel.eval()
    gaitmodel.requires_grad_(False)

    feats = {}

    for inputs in sils:
        try:
            ipts = gaitmodel.module.inputs_pretreament(inputs) \
                if isinstance(gaitmodel, torch.nn.DataParallel) \
                else gaitmodel.inputs_pretreament(inputs)
            
            id = inputs[1][0]
            if id not in feats:
                feats[id] = []
            print(id)
            type = inputs[2][0]
            view = inputs[3][0]

            embs_pkl_path = "{}/{}/{}/{}".format(embs_save_path, id, type, view)
            if not os.path.exists(embs_pkl_path):
                os.makedirs(embs_pkl_path)
            embs_pkl_name = "{}/{}.pkl".format(embs_pkl_path, inputs[3][0])

            with torch.no_grad():
                retval, embs = gaitmodel(ipts)
                torch.cuda.empty_cache()  # 清理显存

            pkl = open(embs_pkl_name, 'wb')
            pickle.dump(embs, pkl)
            feat = {}
            feat[type] = {}
            feat[type][view] = embs
            feats[id].append(feat)
            
        except RuntimeError as e:
            print(f"[OOM Warning] Skipping one sequence due to OOM: {e}")
            torch.cuda.empty_cache()
            continue

    return feats
'''

def gaitfeat_compare(probe_feat:dict, gallery_feat:dict):
    """Compares the feature between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pg_dicts (dict): The id of probe corresponds to the id of gallery
    """
    #获取 probe 特征的键（即 probe 的标识符），然后取第一个作为 probe（假设这里只有一个 probe）
    item = list(probe_feat.keys())
    probe = item[0]

    #pg_dict 用于存储每个 probe 与最匹配的 gallery 的 ID；
    # pg_dicts 存储更多的详细信息（如匹配的 ID 字典）。
    pg_dict = {}
    pg_dicts = {}
    #遍历 probe_feat[probe] 中的每一个元素
    for inputs in probe_feat[probe]:
        number = list(inputs.keys())[0]
        probeid = probe + "-" + number

        galleryid, idsdict = gc.comparefeat(inputs[number]['undefined'], gallery_feat, probeid, 100)
        #galleryid与当前 probe 特征最匹配的 gallery ID。
        #idsdict：详细的匹配信息，可能包含与多个 gallery 特征的匹配程度。
        pg_dict[probeid] = galleryid
        pg_dicts[probeid] = idsdict

    # print("=================== pg_dicts ===================")
    # print(pg_dicts)

    return pg_dict

''' evalueate
def extract_sil(sil, save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        save_path (Path): Output path.
    Returns:
        video_feats (dict): Dictionary of features from the video
    """
    logger.info("begin extracting")
    video_feat = gait_sil(sil, save_path)
    logger.info("extract Done")
    print("video:",video_feat)

    try:
        # 假设 video_feat 是 {'6': [{'001': {'undefined': tensor(...)}]}]
        first_key = next(iter(video_feat))

        first_item = video_feat[first_key][0]

        subject_id = next(iter(first_item))
        view = next(iter(first_item[subject_id]))

        feat_tensor = first_item[subject_id][view]
        return feat_tensor.squeeze().cpu().numpy()
    
    except Exception as e:
        print(f"[Error] Failed to parse feature dict: {e}")
        return None
    #return video_feat
'''

def extract_sil(sil, save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        save_path (Path): Output path.
    Returns:
        video_feats (dict): Dictionary of features from the video
    """
    logger.info("begin extracting")
    video_feat = gait_sil(sil, save_path)
    logger.info("extract Done")
    return video_feat

'''
def extract_sil(sil_path, save_path):
    import time
    import os
    from loguru import logger

    logger.info("begin extracting")
    start_time = time.time()

    video_feat = gait_sil(sil_path, save_path)

    end_time = time.time()
    logger.info("extract Done")
    logger.info(f"[Time] Extracted in {(end_time - start_time):.2f} seconds")

    try:
        # video_feat 结构: {'6': [{'001': {'undefined': tensor(...)}}]}
        video_id = next(iter(video_feat))  # '6'
        subject_dict = video_feat[video_id][0]  # {'001': {...}}
        subject_id = next(iter(subject_dict))  # '001'
        view_dict = subject_dict[subject_id]  # {'undefined': tensor}
        view = next(iter(view_dict))  # 'undefined'
        feat_tensor = view_dict[view]  # tensor([...])

        if isinstance(feat_tensor, torch.Tensor):
            feat_numpy = feat_tensor.squeeze().cpu().numpy()  # shape: (T, D) or (D,)
            print("feat:",feat_numpy)
            return feat_numpy
        else:
            print("[Error] Feature is not a torch.Tensor")
            return None

    except Exception as e:
        print(f"[Error] Failed to extract feature from structure: {e}")
        return None
'''

def compare(probe_feat, gallery_feat):
    """Recognizes  the features between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pgdict (dict): The id of probe corresponds to the id of gallery
    """
    logger.info("begin recognising")
    pgdict = gaitfeat_compare(probe_feat, gallery_feat)
    logger.info("recognise Done")
    print("================= probe - gallery ===================")
    print(pgdict)
    return pgdict

import torch
import torch.nn.functional as F
'''
def cuda_dist(x, y, metric='cosine'):
    """
    x: [n_probe, feat_dim] Tensor
    y: [n_gallery, feat_dim] Tensor
    return: distance matrix [n_probe, n_gallery]
    """
    x = torch.tensor(x, dtype=torch.float32).cuda()
    y = torch.tensor(y, dtype=torch.float32).cuda()

    if metric == 'cosine':
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        dist = 1. - torch.mm(x, y.t())  # cosine distance
    elif metric == 'euclidean':
        dist = torch.cdist(x, y, p=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist

'''
def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

import numpy as np

def compute_mAP(distmat, q_pids, g_pids, q_views=None, g_views=None):
    """
    distmat: [n_probe, n_gallery] distance matrix (numpy)
    q_pids: [n_probe] probe labels
    g_pids: [n_gallery] gallery labels
    q_views: optional, probe views
    g_views: optional, gallery views
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)  # from smallest to largest
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    aps = []
    for i in range(num_q):
        valid = np.ones(num_g, dtype=bool)

        # 排除视角相同（同一视角下的 probe 和 gallery）以避免评估作弊（可选）
        if q_views is not None and g_views is not None:
            valid = valid & (q_views[i] != g_views[indices[i]])

        y_true = matches[i][valid]
        if not np.any(y_true): continue

        y_score = -distmat[i][indices[i]][valid]
        aps.append(average_precision_score(y_true, y_score))

    if len(aps) == 0:
        return 0.0
    return np.mean(aps) * 100  # 百分比形式输出

def average_precision_score(y_true, y_score):
    """
    计算单个 probe 的 AP 值
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)

    indices = np.argsort(-y_score)  # score 从高到低
    y_true = y_true[indices]

    tp = y_true
    fp = 1 - y_true
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    ap = np.sum(precision * y_true) / (np.sum(y_true) + 1e-6)
    return ap
