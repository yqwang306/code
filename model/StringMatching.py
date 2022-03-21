# @Time   : 2022/3/15
# @Author : Yiqiao Wang


"""
String Matching algorithm based on KNN compression for EEG signal classification.
##################################
"""

from DataProcessing.Transform.Base import EEGBaseTransformer
from evaluation.evaluation import *
import Levenshtein
import numpy as np
import pandas as pd
import os


def str_compression(data, k):
    result = ""
    count = 0
    for d in data:
        if d == "0" and count < k:
            count += 1
        else:
            if d != "0" and count > 0:
                result += "0" * count + d
                count = 0
            else:
                count = 0
                result += d
    return result


class StringMatching:
    """:class:`StringMatching`

        Attributes:

        compression_k (int): The maximum number of zero that can appear continuously in a string.

        top_ratio (double): The proportion of the most representative samples.

        test_ratio (double): The proportion for test.

        data_name (str): The name of the dataset.

        data_path (str): The local path of the dataset.

        cases_path (str): The storage path of processed cases data.

        controls_path (str): The storage path of processed controls data.

    """

    def __init__(self, data_name, data_path, compression_k=6, top_ratio=0.3, test_ratio=0.3):
        self.compression_k = compression_k
        self.top_ratio = top_ratio
        self.test_ratio = test_ratio
        self.data_name = data_name
        self.data_path = data_path
        eeg_cases = EEGBaseTransformer(data_name, "bi_class", data_path, 0, 10, True)
        eeg_controls = EEGBaseTransformer(data_name, "bi_class", data_path, 1, 10, True)
        self.cases_path = eeg_cases.generate_data()
        self.controls_path = eeg_controls.generate_data()

    def top_sample(self, top_ratio):
        data_cases = []
        name_cases = []
        data_controls = []
        name_controls = []

        f = open(self.cases_path, 'r', encoding="UTF-8")
        for line in f:
            data_cases.append(line.split(":")[-1])
            name_cases.append(line.split(":")[0])
        f.close()

        acc_cases = []
        for d in data_cases:
            sum = 0
            for ds in data_cases:
                sum += Levenshtein.jaro(d, ds)
            result = sum / data_cases.__len__()
            acc_cases.append(result)

        result = dict(zip(name_cases, acc_cases))
        result = sorted(result.items(), key=lambda x: -x[-1])
        number = int(data_cases.__len__() * top_ratio)

        save_path = "../result/"
        if os.path.exists(save_path) is not True:
            os.makedirs(save_path)

        f = open(save_path + "top_cases_{}.csv".format(self.data_name), "w", encoding="UTF-8")
        first_line = "name,acc\n"
        f.write(first_line)
        for a in range(number):
            result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
            print(result_tmp)
            f.write(result_tmp)
        f.close()

        f = open(self.controls_path, 'r', encoding="UTF-8")
        for line in f:
            data_controls.append(line.split(":")[-1])
            name_controls.append(line.split(":")[0])
        f.close()

        acc_controls = []
        for d in data_controls:
            sum = 0
            for ds in data_controls:
                sum += Levenshtein.jaro(d, ds)
            result = sum / data_controls.__len__()
            acc_controls.append(result)

        result = dict(zip(name_controls, acc_controls))
        result = sorted(result.items(), key=lambda x: -x[-1])
        number = int(data_controls.__len__() * top_ratio)

        f = open(save_path + "top_controls_{}.csv".format(self.data_name), "w", encoding="UTF-8")
        first_line = "name,acc\n"
        f.write(first_line)
        for a in range(number):
            result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
            print(result_tmp)
            f.write(result_tmp)
        f.close()

    def get_top_data(self):
        data_cases = []
        data_controls = []
        path_top_cases = "../result/top_cases_{}.csv".format(self.data_name)
        path_top_controls = "../result/top_controls_{}.csv".format(self.data_name)
        data = pd.read_csv(path_top_cases, sep=',')
        name_top_cases = data["name"].tolist()
        data = pd.read_csv(path_top_controls, sep=',')
        name_top_controls = data["name"].tolist()

        f = open(self.cases_path, 'r', encoding="UTF-8")
        for line in f:
            line_data = line.split(":")
            name = line_data[0]
            if name in name_top_cases:
                data_cases.append(line_data[1])
        f.close()

        f = open(self.controls_path, 'r', encoding="UTF-8")
        for line in f:
            line_data = line.split(":")
            name = line_data[0]
            if name in name_top_controls:
                data_controls.append(line_data[1])
        f.close()

        return data_cases, data_controls

    def calculate_distance_compression(self, top_ratio, test_ratio):
        self.top_sample(top_ratio)
        data_cases_top_tmp, data_controls_top_tmp = self.get_top_data()
        data_cases_top = [str_compression(x, self.compression_k) for x in data_cases_top_tmp]
        data_controls_top = [str_compression(x, self.compression_k) for x in data_controls_top_tmp]

        data_cases = []
        f = open(self.cases_path, 'r', encoding="UTF-8")
        for line in f:
            data_cases.append(str_compression(line.split(":")[-1], self.compression_k))
        f.close()

        data_controls = []
        f = open(self.controls_path, 'r', encoding="UTF-8")
        for line in f:
            data_controls.append(str_compression(line.split(":")[-1], self.compression_k))
        f.close()

        ratio_cases = np.random.randint(0, data_cases.__len__(), int(test_ratio * data_cases.__len__()))
        ratio_controls = np.random.randint(0, data_controls.__len__(), int(test_ratio * data_controls.__len__()))
        print("ratio_cases(count):{}, ratio_controls(count){}".format(ratio_cases.__len__(), ratio_controls.__len__()))
        m = ratio_cases.__len__()
        n = ratio_controls.__len__()

        Detection_queue = [data_cases[x] for x in ratio_cases] + [data_controls[x] for x in ratio_controls]
        result_cases_distant = []
        result_controls_distant = []

        count = 0
        for d in Detection_queue:
            sum = 0
            count += 1
            for sample in data_cases_top:
                sum += Levenshtein.jaro(d, sample)
            result_cases_distant.append(sum / data_cases_top.__len__())
            print("正在处理第{}条数据...".format(count))

        count = 0
        for d in Detection_queue:
            sum = 0
            count += 1
            for sample in data_controls_top:
                sum += Levenshtein.jaro(d, sample)
            result_controls_distant.append(sum / data_controls_top.__len__())
            print("正在处理第{}条数据...".format(count))

        result_cases_distant = np.asarray(result_cases_distant)
        result_controls_distant = np.asarray(result_controls_distant)
        return result_cases_distant, result_controls_distant, m, n

    def run(self):
        result_cases_distant, result_controls_distant, m, n = self.calculate_distance_compression(self.top_ratio, self.test_ratio)

        count_case = 0
        count_control = 0
        tp = fp = fn = tn = 0

        for index in range(m+n):
            if index < m:
                print("cases:", result_cases_distant[index], result_controls_distant[index])
                if result_cases_distant[index] > result_controls_distant[index]:
                    count_case += 1
                    tp += 1
                else:
                    fn += 1
            else:
                print("controls:", result_cases_distant[index], result_controls_distant[index])
                if result_controls_distant[index] > result_cases_distant[index]:
                    count_control += 1
                    tn += 1
                else:
                    fp += 1

        acc_n, acc_p, accuracy, precision, recall = calculate_apr(tp, fp, fn, tn)
        print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)



