import numpy as np
import torch
from utils import *
from networks import *
import logging
from sklearn import metrics
import torch.nn.functional as F


class Trainer(object):
    DEFAULTS = {}
    def __init__(self, model, criterion, optimizer, config, device, use_cuda=True):
        self.__dict__.update(Trainer.DEFAULTS, **config)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = self.dataset
        self.use_cuda = use_cuda
        self.epochs = self.num_epochs
        self.model_save_path = self.model_save_path
        self.device = device
        self.iteration = 0

        if self.use_cuda:
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
        # device_ids = [0, 1]
        # if len(device_ids)>1:
        #     self.model = self.model.to(self.device)
        #     self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        #     self.criterion = self.criterion.to(self.device)

    def vali(self, vali_loader):

        self.model.eval()
        vali_loss = []
        loss_r_list = []
        loss_d_list = []
        for i, (x,y) in enumerate(vali_loader):
            if self.use_cuda:
                x = x.to(self.device )

            loss = 0.0
            input, _, y_true, _ = p_n_sample2(x, self.device)
            input = input.to(self.device )
            y_true = y_true.to(self.device )
            out_r, out_d = self.model(input, 'vali')
            loss_r = F.mse_loss(out_r,input).to(self.device)
            y_true = y_true.to(self.device ).to(torch.float32)
            out_d = out_d.squeeze(1).to(torch.float32).to(self.device )
            loss_d = self.criterion(out_d, y_true)
            loss = loss_r + loss_d
            # loss = loss_r
            vali_loss.append(loss.item())
            loss_r_list.append(loss_r.item())
            loss_d_list.append(loss_d.item())

        print("vali_loss : {:0.4f},, loss_r : {:0.4f},loss_d : {:0.4f},".format(np.average(vali_loss),
                                                                    np.average(loss_r_list),np.average(loss_d_list)))

        return np.average(vali_loss)

    def train(self,train_loader,vali_loader):

        early_stopping = EarlyStopping(self.model_save_path, self.dataset, patience=10)

        for epoch in range(self.epochs):
            loss_list = []
            loss_r_list = []
            loss_d_list = []
            self.model.train()
            print(len(train_loader))
            for i, (x,y) in enumerate(train_loader):
                if self.use_cuda:
                    x =x.to(self.device)

                self.optimizer.zero_grad()
                tra_loss = 0.0
                input, _, y_true, _ = p_n_sample2(x, self.device)
                # print(input.shape)
                input = input.to(self.device )
                y_true = y_true.to(self.device )
                out_r, out_d = self.model(input, 'train')

                loss_r = F.mse_loss(out_r,input).to(self.device)
                y_true = y_true.to(self.device ).to(torch.float32)
                out_d = out_d.squeeze(1).to(torch.float32).to(self.device )
                loss_d = self.criterion(out_d,y_true)
                tra_loss = loss_r + loss_d
                # tra_loss = loss_r
                tra_loss.backward()
                self.optimizer.step()
                loss_list.append(tra_loss.item())
                loss_r_list.append(loss_r.item())
                loss_d_list.append(loss_d.item())
                # print("loss_r : {:0.4f},loss_d : {:0.4f},".format(loss_r,loss_d))
                # print("loss : {:0.4f}".format(tra_loss))

            train_loss = np.average(loss_list)
            loss_r1 = np.average(loss_r_list)
            loss_d1 = np.average(loss_d_list)
            print("epoch = " + str(epoch))
            print("train_loss : {:0.4f}, loss_r : {:0.4f},loss_d : {:0.4f},".format(train_loss,loss_r1,loss_d1))
            vali_loss = self.vali(vali_loader)
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break


#############test###############

    def test(self,test_loader):
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'best_net_test.pth')))
        self.model.eval()
        loss = nn.MSELoss(reduce=False)
        test_labels = []
        test_score = []
        with torch.no_grad():  # 或者@torch.no_grad() 被他们包裹的代码块不需要计算梯度， 也不需要反向传播
            eval_loss = 0.0
            all_pre = []
            all_rec = []
            all_F1 = []

            for i, (x, y) in enumerate(test_loader):
                if self.use_cuda:
                    x = x.to(self.device )
                    y = y.cpu().numpy()

                out,_ = self.model(x, 'test')
                cri = loss(out, x)
                cri = torch.sum(cri, dim=-1)
                cri = cri.detach().cpu().numpy()
                test_score.append(cri)
                test_labels.append(y)

            test_score1 = np.concatenate(test_score, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            score = np.array(test_score1)
            test_labels = np.array(test_labels)

            gt = test_labels.astype(int)
            gt = np.array(gt)

            auc, Fc1, F1_K, max_f1,_,_ = get_metrics(gt, score)

            return auc, Fc1, F1_K, max_f1