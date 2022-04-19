from util.seeg_utils import get_label_data, matrix_normalization

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

class Seegnet(Dataset):  # 任务集的构造
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    resize_x = 130
    resize_y = 200

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, startidx=0):
        """

        :param root: root path of seeg data
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (
            mode, batchsz, n_way, k_shot, k_query))

        # reconstruct input
        if mode != '':
            csvdata, filename_label = self.loadCSV(os.path.join(root, mode))  # csv path
        else:
            csvdata, filename_label = self.loadCSV(root)  # csv path

        self.filename_label = filename_label
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)

    def loadCSV(self, path):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {0: [], 1: []}
        data_labels = get_label_data(path)
        for path, label in data_labels.items():
            dictLabels[label].append(path)
        return dictLabels, data_labels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        # 数据的读取方式, 需要更改为数据集独有的
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 1, self.resize_x, self.resize_y)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 1, self.resize_x, self.resize_y)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [item for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[self.filename_label[item]]
             # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [item for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[self.filename_label[item]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            data = np.load(path)
            result = matrix_normalization(data, (130, 200))
            result = result.astype('float32')
            result = result[np.newaxis, :]
            result = torch.from_numpy(result)
            support_x[i] = result

        query_y_id_list = []
        record_tmp = {}
        for i, path in enumerate(flatten_query_x):
            # 记录query_y的结果
            file_name = os.path.basename(path)
            query_y_id_list.append(file_name)
            ground_truth = self.filename_label[path]

            record_line = {"ground truth": ground_truth, "prediction": ""}
            record_tmp[file_name] = record_line

            # print(record)

            data = np.load(path)

            result = matrix_normalization(data, (130, 200))
            result = result.astype('float32')
            result = result[np.newaxis, :]
            result = torch.from_numpy(result)
            query_x[i] = result
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
        # 需要将文件进行写会
        # file_path = "./precision/{}_val_prediction.pkl".format(patient_test)
        # if os.path.exists(file_path):
        #     record = np.load(file_path, allow_pickle=True)
        #     # print("暂时不进行保存验证！")
        # else:
        #     record = {}
        # for k, v in record_tmp.items():
        #     if k not in record.keys():
        #         record[k] = v
        # with open(file_path, 'wb') as f:
        #     pickle.dump(record, f)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(
            query_y_relative), query_y_id_list

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
