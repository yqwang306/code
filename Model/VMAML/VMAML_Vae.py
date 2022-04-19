from Model.VMAML.Base import BaseMAMLNodel, MyDataset
from util.seeg_utils import matrix_normalization
from Model.VMAML.meta import Meta
from Model.VMAML.MAMLnet import Seegnet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 44, 68
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16,22, 34
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 12, 18
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 10, 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=3),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VMAML_Vae(BaseMAMLNodel):

    def __init__(self, root, processedData, cuda=False, epoch=4000, n_way=2, k_spt=8, k_qry=8, imgsz=100, imgc=5,
                 task_num=5, meta_lr=0.001, update_lr=0.01, update_step=5, update_step_test=10, save_every=None
                 , flag_vae=True, flag_maml=False, plot=False):
        super().__init__(self, root, processedData, cuda, epoch, n_way, k_spt, k_qry, imgsz, imgc,
                 task_num, meta_lr, update_lr, update_step, update_step_test, save_every, flag_vae, flag_maml, plot)

        self.Vae = VAE().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer_vae = torch.optim.Adam(self.Vae.parameters(), lr=0.001) # TODO:vae_lr?

    def trans_data_vae(self, data, label_data, flag):
        shape_data = data.shape
        data_view = data.reshape((-1, self.resize[0], self.resize[1]))
        shape_label = label_data.shape
        label_list = label_data.flatten()
        number = shape_label[0] * shape_label[1]
        result = []
        loss_all = 0.0

        for i in range(number):
            data_tmp = data_view[i]
            data_tmp = matrix_normalization(data_tmp, (128, 200))
            data_tmp = data_tmp[np.newaxis, np.newaxis, :]
            data_tmp = torch.from_numpy(data_tmp)
            data_tmp = data_tmp.to(self.device)
            recon_batch = self.Vae(data_tmp)
            loss_all += self.criterion(recon_batch, data_tmp)

            result_tmp = recon_batch.detach().cpu().numpy()
            result_tmp = result_tmp.reshape((128, 200))
            result_tmp = matrix_normalization(result_tmp, self.resize)
            data_result = result_tmp[np.newaxis, :]
            result.append(data_result)

        if flag:  # 交替训练的控制条件
            self.optimizer_vae.zero_grad()
            loss_all.backward()
            self.optimizer_vae.step()
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
                       k_query=self.k_qry, batchsz=self.epoch)
        mini_test = Seegnet(os.path.join(self.root, 'split'), mode='test', n_way=self.n_way, k_shot=self.k_spt,
                            k_query=self.k_qry, batchsz=100)
        last_accuracy = 0.0
        plt_train_loss = []
        plt_train_acc = []

        plt_test_loss = []
        plt_test_acc = []

        for epoch in tqdm(range(self.epoch)):  # 设置迭代次数
            # fetch meta_batchsz num of episode each time
            db = DataLoader(mini, self.task_num, shuffle=True, num_workers=1, pin_memory=True)

            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
                x_spt_vae, loss_spt = self.trans_data_vae(x_spt.numpy(), y_spt)
                x_qry_vae, loss_qry = self.trans_data_vae(x_qry.numpy(), y_qry)
                x_spt_vae = torch.from_numpy(x_spt_vae)
                x_qry_vae = torch.from_numpy(x_qry_vae)
                x_spt_vae, y_spt, x_qry_vae, y_qry = x_spt_vae.to(self.device), y_spt.to(self.device), x_qry_vae.to(
                    self.device), y_qry.to(self.device)

                accs, loss_q = self.maml(x_spt_vae, y_spt, x_qry_vae, y_qry, self.flag_maml)

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
                        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                        accs_all_test = []
                        loss_all_test = []

                        for x_spt, y_spt, x_qry, y_qry in db_test:
                            x_spt_vae, loss_spt = self.trans_data_vae(x_spt.numpy(), y_spt, self.flag_vae)
                            x_qry_vae, loss_qry = self.trans_data_vae(x_qry.numpy(), y_qry, self.flag_vae)
                            x_spt_vae = torch.from_numpy(x_spt_vae)
                            x_qry_vae = torch.from_numpy(x_qry_vae)
                            x_spt_vae, y_spt, x_qry_vae, y_qry = x_spt_vae.squeeze(0).to(self.device), y_spt.squeeze(0).to(
                                self.device), x_qry_vae.squeeze(0).to(self.device), y_qry.squeeze(0).to(self.device)

                            accs, loss_test = self.maml.finetunning(x_spt_vae, y_spt, x_qry_vae, y_qry)
                            loss_all_test.append(loss_spt.item() + loss_qry.item() + loss_test.item())
                            accs_all_test.append(accs)

                        # [b, update_step+1]
                        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                        plt_test_acc.append(accs[-1])
                        avg_loss = np.mean(np.array(loss_all_test))
                        plt_test_loss.append(avg_loss)

                    print('Test acc:', accs)
                    test_accuracy = accs[-1]

                    if test_accuracy >= last_accuracy:
                        # save networks
                        torch.save(self.maml.state_dict(), str(
                            "./models/maml" + str(self.n_way) + "way_" + str(self.k_spt) + "shot.pkl"))
                        last_accuracy = test_accuracy

                        torch.save(self.Vae.state_dict(), "./models/Vae.pkl")
                        print("{} and {} model have saved!!!".format("maml", "vae"))

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