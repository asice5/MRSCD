import numpy as np
import torch
import random
from itertools import combinations
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def p_n_sample2(x, device):
    s1,_,y1,_ = p_n_sample(x,8,0.6,20,device)
    s2, _,y2, _ = p_n_sample(x, 8, 0.6, 5, device)
    # s3, _,y3, _ = p_n_sample(x, 2, 0.6, 2, device)

    # s = torch.cat((s1, s2), 0).to(device)
    sample_all = torch.cat((s1, s2), 0).to(device)
    # y = torch.cat((y1, y2), 0).to(device)
    y_true = torch.cat((y1, y2), 0).to(device)
    return (sample_all, sample_all.shape, y_true, y_true.shape)

def p_n_sample(x, sample_num, samp_rate, s_win_num, device):
    s_win_size = int(x.shape[1] // s_win_num)
    a_1 = torch.stack(torch.split(x, s_win_size, 1), 1)  ###shape=[batchsize,s_win_num=10,s_win_size,f]
    batchsize = x.shape[0]
    sample_num = sample_num
    samp_rate = samp_rate
    s_win_num = s_win_num
    postive_sample_1_list = []
    postive_sample_2_list = []
    negtive_sample_list = []

    for i in range(batchsize):
        for j in range(sample_num):
            idxn = random.sample(range(s_win_num), int(s_win_num * samp_rate))  ####shape[6,1]
            idxn2 = random.sample(range(s_win_num), int(s_win_num * samp_rate))
            idxn3 = random.sample(range(s_win_num), int(s_win_num * samp_rate))
            idxn1 = sorted(idxn)
            idxn2 = sorted(idxn2)
            idxn3 = sorted(idxn3)
            idxp1 = idxn1[:]
            idxp2 = idxn2[:]
            idxn4 = idxn3[:]
            random.shuffle(idxn3)
            while (idxn3 == idxn4):
                random.shuffle(idxn3)
            postive_sample_1_list.append(a_1[i, idxp1, :, :].clone().detach())
            postive_sample_2_list.append(a_1[i, idxp2, :, :].clone().detach())
            negtive_sample_list.append(a_1[i, idxn3, :, :].clone().detach())
    postive_sample_1 = torch.stack(postive_sample_1_list).to(device)  ###shape[2*sample_num, 6*s_win_size, f]
    postive_sample_2 = torch.stack(postive_sample_2_list).to(device)
    negtive_sample_all = torch.stack(negtive_sample_list).to(device)
    postive_sample_1 = torch.reshape(postive_sample_1, (postive_sample_1.shape[0], -1, postive_sample_1.shape[3])).to(device)
    postive_sample_2 = torch.reshape(postive_sample_2, (postive_sample_2.shape[0], -1, postive_sample_2.shape[3])).to(device)
    negtive_sample_all = torch.reshape(negtive_sample_all,
                                           (negtive_sample_all.shape[0], -1, negtive_sample_all.shape[3])).to(device)

    posttive_cp = torch.cat((postive_sample_1.unsqueeze(1), postive_sample_2.unsqueeze(1)), 1).to(device)###shape[sample_num,2,6*s_win_size, f]
    negtive_cp = torch.cat((postive_sample_1.unsqueeze(1), negtive_sample_all.unsqueeze(1)), 1).to(device)
    negtive_cp1 = torch.cat((postive_sample_2.unsqueeze(1), negtive_sample_all.unsqueeze(1)), 1).to(device)
    sample_all1 = torch.cat((posttive_cp.reshape(posttive_cp.shape[0], -1, posttive_cp.shape[3]),
                                negtive_cp.reshape(negtive_cp.shape[0], -1, negtive_cp.shape[3])), 0).to(device)
    sample_all = torch.cat((sample_all1,negtive_cp1.reshape(negtive_cp1.shape[0], -1, negtive_cp1.shape[3])), 0).to(device)
    sample_all = sample_all.reshape(-1, sample_all.shape[1] // 2, sample_all.shape[2]).to(device)
    y_p = torch.zeros(sample_all1.shape[0] // 2).to(device)
    y_n = torch.ones(sample_all1.shape[0] // 1).to(device)
    y_true = torch.cat((y_p, y_n), 0).to(device)

    return (sample_all, sample_all.shape, y_true, y_true.shape)

class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, num_hiddens, input_size, output_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # # 定义RNN层
        # 输入的形状为（num_steps, batch_size, input_size）  # input_size 就是 vocab_size
        # 输出的形状为（num_steps, batch_size, num_hiddens）
        self.rnn = nn.GRU(input_size, num_hiddens,2)
        self.input_size = self.rnn.input_size
        self.num_hiddens = self.rnn.hidden_size
        self.output_size = output_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.output_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.input_size)

    def forward(self, inputs, state):
        # inputs的形状为（num_steps, batch_size, input_size）
        # Y是所有时间步的隐藏状态，state是最后一个时间步的隐藏状态
        # Y的形状为（num_steps, batch_size, hidden_size），state为（1，batch_size, hidden_size）
        Y, state = self.rnn(inputs, state)
        Y = self.linear(Y)
        out = Y.transpose(0, 1)
        return out, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))




class MRSCD(torch.nn.Module):
    def __init__(self, input_gru_size, num_hiddens, input_fea_size, win_size_test, output_gru_size, device):
        super(MRSCD, self).__init__()
        self.device = device
        self.fea_size = input_fea_size
        self.num_hiddens = num_hiddens
        self.output_gru_size = output_gru_size
        self.win_size_test = win_size_test
        self.linear_em = nn.Linear(self.fea_size, input_gru_size)
        self.gru = RNNModel(self.num_hiddens, input_gru_size, self.output_gru_size).to(self.device)###Y的形状为（num_steps, batch_size, hidden_size）,之后有全连接

        self.projection_head_c = nn.Sequential(
            nn.Linear(self.output_gru_size, 40),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(40, self.fea_size),)

        # self.projection_head_d = nn.Sequential(
        #     nn.Linear(2*self.win_size_test*self.output_gru_size, 32),
        #     nn.BatchNorm1d(32),
        #     nn.Tanh(),
        #     # nn.Sigmoid(),
        #     nn.Linear(32, 1), )
        self.projection_head_d = nn.Sequential(
            nn.Linear(2*self.win_size_test*self.fea_size, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            # nn.Sigmoid(),
            nn.Linear(32, 1), )


    def forward(self, x, mode):

      ####shape=[(p_num+n_num)*batchsize*2, t_num, f]
        out_em = self.linear_em(x)

        state = self.gru.begin_state(self.device,out_em.shape[0])
        out_em = out_em.transpose(0,1)
        out_gru,_ = self.gru(out_em.to(self.device),state)        ####shape=[(p_num+n_num)*batchsize*2,t_num, 20]
        # out_ind = out_gru.reshape((out_gru.shape[0],-1))

        out_c = self.projection_head_c(out_gru)      ####shape[(p_num+n_num)*batchsize*2,t_num,?]

        if mode == 'test':

            return out_c, _
        else:
            out_ind = out_c.reshape((out_c.shape[0], -1))
            out_ph = out_ind.reshape((-1, 2, out_ind.shape[-1]))  ####shape[(p_num+n_num)*batchsize, 2, t_num*?]
            input_d = out_ph.reshape((out_ph.shape[0], -1))  ####shape[(p_num+n_num)*batchsize, 64]
            out_d = self.projection_head_d(input_d)  ####shape[(p_num+n_num)*batchsize,1]

            out_d = torch.sigmoid(out_d)

            return out_c, out_d



# class cc(torch.nn.Module):
#     def __init__(self, input_gru_size, num_hiddens, input_fea_size, win_size_test, output_gru_size, device):
#         super(cc, self).__init__()
#         self.device = device
#         self.fea_size = input_fea_size
#         self.num_hiddens = num_hiddens
#         self.output_gru_size = output_gru_size
#         self.win_size_test = win_size_test
#         self.linear_em = nn.Linear(self.fea_size, input_gru_size)
#         self.gru = RNNModel(self.num_hiddens, input_gru_size, self.output_gru_size).to(self.device)###Y的形状为（num_steps, batch_size, hidden_size）,之后有全连接

#         self.projection_head_c = nn.Sequential(
#             nn.Linear(self.output_gru_size, 40),
#             # nn.ReLU(inplace=True),
#             nn.Tanh(),
#             nn.Linear(40, self.fea_size),)

#         # self.projection_head_d = nn.Sequential(
#         #     nn.Linear(2*self.win_size_test*self.output_gru_size, 32),
#         #     nn.BatchNorm1d(32),
#         #     nn.Tanh(),
#         #     # nn.Sigmoid(),
#         #     nn.Linear(32, 1), )
#         self.projection_head_d = nn.Sequential(
#             nn.Linear(2*self.win_size_test*self.fea_size, 1),)
#             # nn.BatchNorm1d(32),
#             # nn.Tanh(),
#             # # nn.Sigmoid(),
#             # nn.Linear(32, 1), )


#     def forward(self, x, mode):

#       ####shape=[(p_num+n_num)*batchsize*2, t_num, f]
#         out_em = self.linear_em(x)

#         state = self.gru.begin_state(self.device,out_em.shape[0])
#         out_em = out_em.transpose(0,1)
#         out_gru,_ = self.gru(out_em.to(self.device),state)        ####shape=[(p_num+n_num)*batchsize*2,t_num, 20]
#         # out_ind = out_gru.reshape((out_gru.shape[0],-1))

#         out_c = self.projection_head_c(out_gru)      ####shape[(p_num+n_num)*batchsize*2,t_num,?]

#         if mode == 'test':

#             return out_c, _
#         else:
#             out_ind = out_c.reshape((out_c.shape[0], -1))
#             out_ph = out_ind.reshape((-1, 2, out_ind.shape[-1]))  ####shape[(p_num+n_num)*batchsize, 2, t_num*?]
#             input_d = out_ph.reshape((out_ph.shape[0], -1))  ####shape[(p_num+n_num)*batchsize, 64]
#             out_d = self.projection_head_d(input_d)  ####shape[(p_num+n_num)*batchsize,1]

#             out_d = torch.sigmoid(out_d)

#             return out_c, out_d