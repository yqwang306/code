# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 15.03.2022

from Model.VMAML.Base import BaseMAMLNodel
from Model.VMAML.meta import Meta
from Model.VMAML.MAMLnet import Seegnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os

class VAE(nn.Module): # customized the structure of VAE
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(self.resize[0] * self.resize[1], 1000)
        self.fc21 = nn.Linear(1000, 20)
        self.fc22 = nn.Linear(1000, 20)
        self.fc3 = nn.Linear(20, 1000)
        self.fc4 = nn.Linear(1000, self.resize[0] * self.resize[1])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.resize[0] * self.resize[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VMAML_Double_Vae(BaseMAMLNodel):

    def __init__(self, root, processedData, cuda=False, epoch=4000, n_way=2, k_spt=5, k_qry=5, imgsz=500, imgc=5,
                 task_num=8, meta_lr=0.001, update_lr=0.01, update_step=5, update_step_test=8, save_every=None, vae_lr=0.01,
                 flag_vae=True, flag_maml=False, plot=False):
        super().__init__(self, root, processedData, cuda, epoch, n_way, k_spt, k_qry, imgsz, imgc,
                 task_num, meta_lr, update_lr, update_step, update_step_test, save_every, flag_vae, flag_maml, plot)

        self.vae_lr = vae_lr # TODO: never used?
        self.vae_p = VAE().to(self.device)
        self.vae_n = VAE().to(self.device)
        self.optimizer_vae_p = torch.optim.Adam(self.vae_p.parameters(), lr=0.002)
        self.optimizer_vae_n = torch.optim.Adam(self.vae_n.parameters(), lr=0.002)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.resize[0] * self.resize[1]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def trans_data_vae(self, data, label_data):
        shape_data = data.shape
        data_view = data.reshape((-1, self.resize[0], self.resize[1]))
        shape_label = label_data.shape
        label_list = label_data.flatten()
        number = shape_label[0] * shape_label[1]
        result = []
        loss_all = 0.0

        for i in range(number):
            data_tmp = data_view[i]
            data_tmp = torch.from_numpy(data_tmp)
            data_tmp = data_tmp.to(self.device)

            if label_list[i] == 1:  # positive
                self.optimizer_vae_p.zero_grad()
                recon_batch, mu, logvar = self.vae_p(data_tmp)
                loss = self.loss_function(recon_batch, data_tmp, mu, logvar)
                loss.backward()
                self.optimizer_vae_p.step()
            else:
                self.optimizer_vae_n.zero_grad()
                recon_batch, mu, logvar = self.vae_n(data_tmp)
                loss = self.loss_function(recon_batch, data_tmp, mu, logvar)
                loss.backward()
                self.optimizer_vae_n.step()
            loss_all += loss.item()
            result_tmp = recon_batch.detach().cpu().numpy()
            result_tmp = result_tmp.reshape(self.resize)
            data_result = result_tmp[np.newaxis, :]
            result.append(data_result)
        result_t = np.array(result)
        result_r = result_t.reshape(shape_data)
        loss_all = loss_all / number

        return result_r, loss_all

    def train_model(self):
        torch.manual_seed(222)  # 为cpu设置种子，为了使结果是确定的
        torch.cuda.manual_seed_all(222)  # 为GPU设置种子，为了使结果是确定的
        np.random.seed(222)

        # import vae module
        args = {'update_lr': self.update_lr, 'meta_lr': self.meta_lr, 'n_way': self.n_way,
                'k_spt': self.k_spt, 'k_qry': self.k_qry, 'task_num': self.task_num,
                'update_step': self.update_step, 'update_step_test': self.update_step_test}
        self.maml = Meta(args, self.config).to(self.device)
        tmp = filter(lambda x: x.requires_grad, self.maml.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(self.maml)
        print('Total trainable tensors:', num)

        # batchsz here means total episode number
        mini = Seegnet(os.path.join(self.root, 'split'), mode='train', n_way=self.n_way, k_shot=self.k_spt,
                       k_query=self.k_qry,
                       batchsz=self.epoch)
        mini_test = Seegnet(os.path.join(self.root, 'split'), mode='test', n_way=self.n_way, k_shot=self.k_spt,
                            k_query=self.k_qry,
                            batchsz=100)
        last_accuracy = 0.0
        plt_train_loss = []
        plt_train_acc = []

        plt_test_loss = []
        plt_test_acc = []

        for epoch in tqdm(range(self.epoch)):  # 设置迭代次数
            # fetch meta_batchsz num of episode each time
            db = DataLoader(mini, self.task_num, shuffle=True, pin_memory=True)

            for step, (x_spt, y_spt, x_qry, y_qry, _) in enumerate(tqdm(db)):
                x_spt_vae, loss_spt = self.trans_data_vae(x_spt.numpy(), y_spt)
                x_qry_vae, loss_qry = self.trans_data_vae(x_qry.numpy(), y_qry)
                x_spt_vae = torch.from_numpy(x_spt_vae)
                x_qry_vae = torch.from_numpy(x_qry_vae)
                x_spt_vae, y_spt, x_qry_vae, y_qry = x_spt_vae.to(self.device), y_spt.to(self.device), x_qry_vae.to(
                    self.device), y_qry.to(self.device)

                accs, loss_q = self.maml(x_spt_vae, y_spt, x_qry_vae, y_qry, self.resize)

                if self.save_every != None and step % self.save_every == 0:
                    model_path = "./models/maml{}way_{}shot_epoch_{}.pkl".format(self.n_way, self.k_spt, step)
                    torch.save(self.maml.state_dict(), model_path)
                    print("epoch {} model has been saved!".format(step))

                if step % 20 == 0:  # TODO: update step?
                    d = loss_q.cpu()
                    dd = d.detach().numpy()
                    plt_train_loss.append(dd)
                    plt_train_acc.append(accs[-1])
                    print('step:', step, '\ttraining acc:', accs)

                    if step % 50 == 0:  # evaluation  # TODO: update step test?
                        db_test = DataLoader(mini_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
                        accs_all_test = []
                        loss_all_test = []

                        for x_spt, y_spt, x_qry, y_qry, _ in db_test:
                            x_spt_vae, loss_spt = self.trans_data_vae(x_spt.numpy(), y_spt)
                            x_qry_vae, loss_qry = self.trans_data_vae(x_qry.numpy(), y_qry)
                            x_spt_vae = torch.from_numpy(x_spt_vae)
                            x_qry_vae = torch.from_numpy(x_qry_vae)
                            x_spt_vae, y_spt, x_qry_vae, y_qry = x_spt_vae.squeeze(0).to(self.device), y_spt.squeeze(0).to(
                                self.device), x_qry_vae.squeeze(0).to(self.device), y_qry.squeeze(0).to(self.device)

                            result, loss_test = self.maml.finetunning(x_spt_vae, y_spt, x_qry_vae, y_qry)
                            acc = result['accuracy']

                            loss_all_test.append(loss_test.item())
                            accs_all_test.append(acc)

                        # [b, update_step+1]
                        # accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                        plt_test_acc.append(acc)
                        avg_loss = np.mean(np.array(loss_all_test))
                        plt_test_loss.append(avg_loss)

                    test_accuracy = np.array(accs_all_test).mean()
                    print('Test acc:', test_accuracy)

                    if test_accuracy > last_accuracy:
                        # save networks
                        torch.save(self.maml.state_dict(), str("./models/maml" + str(self.n_way) + "way_" + str(self.k_spt) + "shot.pkl"))
                        last_accuracy = test_accuracy

                        torch.save(self.vae_p.state_dict(), "./models/Vae_positive.pkl")
                        print("VAE positive model saved successfully!")
                        torch.save(self.vae_n.state_dict(), "./models/Vae_negative.pkl")
                        print("VAE negative model saved successfully!")
                        print("{} and {} models have been saved!!!".format("maml", "vae"))

        if self.plot:
            plt.figure(1)
            plt.title("testing info")
            plt.xlabel("episode")
            plt.ylabel("Acc/loss")
            plt.plot(plt_test_loss, label='Loss')
            plt.plot(plt_test_acc, label='Acc')
            plt.legend(loc='upper right')
            plt.savefig('./drawing/test.png')
            plt.show()

            plt.figure(2)
            plt.title("training info")
            plt.xlabel("episode")
            plt.ylabel("Acc/loss")
            plt.plot(plt_train_loss, label='Loss')
            plt.plot(plt_train_acc, label='Acc')
            plt.legend(loc='upper right')
            plt.savefig('./drawing/train.png')
            plt.show()