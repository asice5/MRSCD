import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random
from torch.backends import cudnn
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, dataset, patience=20, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.dataset = dataset
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path,'best_net_test.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark=  False

# def log_sum_exp(x):
#     x_max = x.data.max()
#     return torch.log(torch.sum(torch.exp(x-x_max)))

# def cc_loss(y_true, out_c,device,temperature = 0.5):
#
#     # out_c = torch.sigmoid(out_c).to(device)
#     out_c = out_c.reshape(-1, 2, out_c.shape[1])
#     out_c = F.normalize(out_c,p=2,dim=2)
#     idx_p = torch.nonzero(y_true > 0, as_tuple=False).to(device)
#     postive = out_c[idx_p.squeeze(),:,:]
#     idx_n = torch.nonzero(y_true < 1, as_tuple=False).to(device)
#     negtive = out_c[idx_n.squeeze(),:,:]
#     postive = postive.reshape(-1, postive.shape[2])
#     negtive = negtive.reshape(-1, negtive.shape[2])
#     postive_even = postive[::2,:]
#     postiv_odd = postive[1::2,:]
#     negtive_even = negtive[::2,:]
#     negtive_odd = negtive[1::2,:]
#     # 计算相似度
#
#     p_similarities = F.cosine_similarity(postive_even, postiv_odd,dim=1)
#     n_similarities = F.cosine_similarity(negtive_even, negtive_odd,dim=1)
#
#     loss_p = torch.exp(p_similarities.sum(dim = 0)/(temperature *1000))  ####值过大
#     # loss_p =torch.logsumexp(p_similarities,0)
#     # loss_n = torch.logsumexp(n_similarities,0 )
#     loss_n = torch.exp(n_similarities.sum() / (temperature *1000))
#     loss = -torch.log(loss_p/(loss_n + loss_p))
#     # loss = -torch.log(loss_p / loss_n)
#
#     return loss

def get_metrics(label, score):
    auc = roc_auc_score(label, score)

    events, i = [], 0
    while i < label.shape[0]:
        if label[i] == 1:
            start = i
            while i < label.shape[0] and label[i] == 1:
                i += 1
            end = i
            events.append((start, end))
        else:
            i += 1

    Fc1, F1_K, max_f1, _P, _R = eval_result(label, score, events)
    return auc, Fc1, F1_K, max_f1, _P, _R

def eval_result(label, test_scores,  events):
    max_Fc1 = 0.0
    max_F1_K = 0.0
    max_f1, _P, _R = 0.0, 0.0, 0.0
    for ratio in np.arange(0, 50.1, 0.1):
        threshold = np.percentile(test_scores, 100 - ratio)
        pred = (test_scores > threshold).astype(int)
        Fc1 = cal_Fc1(pred, events, label)
        if Fc1 > max_Fc1:
            max_Fc1 = Fc1

        F1_K = []
        for K in np.arange(0, 1.1, 0.1):
            F1_K.append(cal_F1_K(K, pred.copy(), events, label))
        AUC_F1_K = np.trapz(np.array(F1_K), np.arange(0, 1.1, 0.1))
        if AUC_F1_K > max_F1_K:
            max_F1_K = AUC_F1_K

        f1, P, R = cal_F1(pred.copy(), events, label)
        if f1 > max_f1:
            max_f1 = f1
            _P = P
            _R = R
    return max_Fc1, max_F1_K, max_f1, _P, _R


def cal_Fc1(pred, events, label):
    tp = np.sum([pred[start:end].any() for start, end in events])
    fn = len(events) - tp
    rec_e = tp / (tp + fn)
    prec_t = precision_score(label, pred)
    if prec_t == 0 and rec_e == 0:
        Fc1 = 0
    else:
        Fc1 = 2 * rec_e * prec_t / (rec_e + prec_t)
    return Fc1


def cal_F1_K(K, pred, events, label):
    for start, end in events:
        if np.sum(pred[start:end]) > K * (end - start):
            pred[start:end] = 1
    return f1_score(label, pred)


def cal_F1(pred, events, label):
    pred = detection_adjustment(pred, events)
    f1 = f1_score(label, pred)
    P = precision_score(label, pred)
    R = recall_score(label, pred)
    return f1, P, R


def detection_adjustment(pred, events):
    for start, end in events:
        if np.sum(pred[start:end]) > 0:
            pred[start:end] = 1
    return np.array(pred)