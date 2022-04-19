# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 15.03.2022

from Model.Base import BaseModel
from Model.VMAML.meta import Meta
from util.seeg_utils import matrix_normalization
from util.common_utils import IndicatorCalculation

from torch.utils.data import DataLoader
from scipy.special import softmax
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import os
import numpy as np

class BaseMAMLNodel(BaseModel, ABC):

    def __init__(self, root, processedData, cuda=False, epoch=4000, n_way=2, k_spt=8, k_qry=8, imgsz=100, imgc=5,
                 task_num=5, meta_lr=0.001, update_lr=0.01, update_step=5, update_step_test=10, save_every=None,
                 flag_vae=True, flag_maml=True, plot=False):
        self.cuda = cuda and torch.cuda.is_available()
        self.data = processedData # SeegData(),well processed Data after DataProcessing
        self.root = root
        self.epoch = epoch # epoch number
        self.n_way = n_way
        self.k_spt = k_spt # k shot for support set
        self.k_qry = k_qry # k shot for query set
        self.imgsz = imgsz
        self.imgc = imgc
        self.task_num = task_num # meta batch size, namely task num
        self.meta_lr = meta_lr # meta-level outer learning rate
        self.updata_lr = update_lr # task-level inner update learning rate
        self.update_step = update_step # task-level inner update steps
        self.update_step_test = update_step_test # update steps for finetunning
        self.save_every = save_every # save model every a certain number of steps
        self.flag_vae = flag_vae
        self.flag_maml = flag_maml
        self.plot = plot

        self.train_path = os.path.join(root, 'split/train')
        self.test_path = os.path.join(root, 'split/test')
        self.val_path = os.path.join(root, 'split/val')
        self.device = torch.device("cuda" if self.cuda else "cpu")
        # define the structure of the Neural Network Model
        self.config = [
            ('conv2d', [32, 1, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            ('flatten', []),
            ('linear', [self.n_way, 7040])]
        self.resize = (130, 200)

    def load_model(self, model_path):
        args = {'update_lr': self.update_lr, 'meta_lr': self.meta_lr, 'n_way': self.n_way,
                'k_spt': self.k_spt, 'k_qry': self.k_qry, 'task_num': self.task_num,
                'update_step': self.update_step, 'update_step_test': self.update_step_test}
        self.maml = Meta(args, self.config).to(self.device)
        if os.path.exists(model_path):
            self.maml.load_state_dict(torch.load(model_path))
        else:
            print("model is not exist!")

    def predict_model(self, data_loader):
        predict = []

        for data, label, name_id in data_loader: # TODO: may have problems cuz patient_name doesn't be stored anymore
            data = data.to(self.device)
            with torch.no_grad():
                result = self.maml.net(data)
                c_result = result.cpu().detach().numpy()
                r = softmax(c_result, axis=1)
                pre_y = r.argmax(1)
                predict.append(pre_y)

        return predict

    def evaluate_model(self, model_path):
        # load in model
        self.maml = self.load_model(model_path)
        # load in val data
        data_val = []
        for (index, d) in enumerate(os.listdir(self.val_path)):
            path_l = os.path.join(self.val_path, d)
            names = os.listdir(path_l)
            data_val += [(os.path.join(path_l, x), index, x) for x in names]
        my_dataset = MyDataset(data_val)
        data_loader = DataLoader(my_dataset, batch_size=1, shuffle=True)

        pre_result = {}
        pre_list = []
        true_list = []

        for data, label, name_id in data_loader: # TODO: may have problems cuz patient_name doesn't be stored anymore
            data = data.to(self.device)
            with torch.no_grad():
                result = self.maml.net(data)
                c_result = result.cpu().detach().numpy()
                r = softmax(c_result, axis=1)
                pre_y = r.argmax(1)
                pre_result[name_id] = pre_y
                pre_list.append(pre_y[0])
                true_list.append(label[0])
        cal = IndicatorCalculation()
        cal.set_values(pre_list, true_list)
        print("Accuracy:{}, Precision:{}, Recall:{}, f1_score:{}".format(cal.get_accuracy(), cal.get_precision(),
                                                                         cal.get_recall(), cal.get_f1score()))

    @abstractmethod
    def trans_data_vae(self):
        # insert VAE Module into different models, need to be rewrite every time
        pass

    @abstractmethod
    def maml_framework(self):
        # using MAML to train the most optimal initial parameters
        pass

class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, index):
        fn, label, name_id = self.imgs[index]
        data = np.load(fn)
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        return result, label, name_id # TODO: may have problems cuz patient_name doesn't be stored anymore

    def __len__(self):
        return len(self.imgs)