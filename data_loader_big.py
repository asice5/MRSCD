import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class trainDataLoader(Dataset):
    def __init__(self, data_name, data_path, win_size, step, mode, scaler):
        self.name = data_name
        self.win_size = win_size
        self.step = step
        self.mode = mode

        self.scaler = StandardScaler()
        if scaler == 'M':
            self.scaler = MinMaxScaler()

        allow_pickle = False
        if self.name == "SWAT":
            allow_pickle = True

        train = np.load(data_path + data_name + "/" + data_name + "_train.npy", allow_pickle=allow_pickle)
        self.scaler.fit(train)
        train = self.scaler.transform(train)

        self.train = train[:math.floor(train.shape[0] * 0.8)]
        self.val = train[math.floor(train.shape[0] * 0.8):]
        self.n_features = self.train.shape[1]

        test = np.load(data_path + data_name + "/" + data_name + "_test.npy", allow_pickle=allow_pickle)
        self.test = self.scaler.transform(test)
        self.test_label = np.load(data_path + data_name + "/" + data_name + "_test_label.npy",
                                  allow_pickle=allow_pickle)

    def __len__(self):
        if self.mode == "train":
            return max((self.train.shape[0] - self.win_size) // self.step + 1, 0)
        elif self.mode == 'val':
            return max((self.val.shape[0] - self.win_size) // self.step + 1, 0)
        else:
            return max((self.test.shape[0] - self.win_size) // self.step + 1, 0)

    def __getitem__(self, index):
        index *= self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.empty(1)
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.empty(1)
        else:
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_label[index:index + self.win_size])




def get_segment_data_loader(data_name, data_path, win_size, scaler, step, mode, batch_size,shuffle=False):

    dataset = trainDataLoader(data_name, data_path, win_size, step, mode, scaler)
    if mode == 'train' :
        shuffle = True
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset.n_features, data_loader


# _, test_post_loader = get_segment_data_loader(data_name='SWAT', data_path='./Data/Data/',
#                                                   win_size=10, scaler='M', step=5,
#                                                  mode='train',batch_size=2,shuffle=True)
#
#
# for i,(x) in enumerate(test_post_loader):
#     a = x
#     print(x,x.shape)
#     break
# # print(a,a.shape)
