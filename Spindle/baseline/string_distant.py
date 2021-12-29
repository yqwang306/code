from Spindle.preprocessing.preprocessing import SpindleData
import Levenshtein
import numpy as np
import pandas as pd
from Spindle.evaluation.calculate_class_info import CA


# 用于统计相关信息
def calculate_distance(run_path, ratio=0.2):  # 计算距离的评价标准是和样本字符串进行比较(非压缩的版本)
    # -------------------------------------------基于全部数据的存储比较-------------------------------------------
    f = open(run_path +"/cases_encoding_str.txt", 'r', encoding="UTF-8")
    data_cases = []
    for line in f:
        data_cases.append(line.split(":")[-1])
    print("cases_encoding_str文件读取完成！")
    f.close()
    data_controls = []
    f = open(run_path + "/controls_encoding_str.txt", 'r', encoding="UTF-8")
    for line in f:
        data_controls.append(line.split(":")[-1])
    print("controls_encoding_str文件读取完成！")
    f.close()
    data_cases_top = data_cases
    data_controls_top = data_controls  # 只是为了保持形式的一致性

    # --------------------------------------------选择基本的数据---------------------------------------------

    ratio_cases = np.random.randint(0, data_cases.__len__(), int(ratio * data_cases.__len__()))  # 选取20%进行测试
    ratio_control = np.random.randint(0, data_controls.__len__(), int(ratio * data_controls.__len__()))
    print("ratio_cases(count):{}, ratio_controls(count){}".format(ratio_cases.__len__(), ratio_control.__len__()))
    m = ratio_cases.__len__()
    n = ratio_control.__len__()

    Detection_queue = [data_cases[x] for x in ratio_cases] + [data_controls[x] for x in
                                                              ratio_control]  # 测试集，既包含cases也包含controls
    result_cases_distant = []
    result_controls_distant = []
    count = 0
    for d in Detection_queue:  # 记录病人的信息
        sum = 0
        count += 1  # 计算测试集中样本个数
        for sample in data_cases_top:  # 这里data_cases_top是全部的data_cases
            sum += Levenshtein.jaro(d, sample)  # 即计算测试集中的每个样本与P(失眠)中的一组选定序列的相似度
        result_cases_distant.append(sum / data_cases_top.__len__())
        print("正在处理第{}条数据...".format(count))
    count = 0
    for d in Detection_queue:
        sum = 0
        count += 1
        for sample in data_controls_top:
            sum += Levenshtein.jaro(d, sample)  # 即计算测试集中的每个样本与N(正常)中的一组选定序列的相似度
        result_controls_distant.append(sum / data_controls_top.__len__())
        print("正在处理第{}条数据...".format(count))

    result_cases_distant = np.asarray(result_cases_distant)
    result_controls_distant = np.asarray(result_controls_distant)
    dim = len(data_controls[0])  # 即controls中的字符串长度
    count_case = 0
    count_control = 0

    tp = fp = fn = tn = 0

    for index in range(result_controls_distant.__len__()):
        if index < m:  # 实际结果为cases
            if result_cases_distant[index] > result_controls_distant[index]:  # 与cases序列的相似度较大，说明分类正确
                count_case += 1
                tp += 1  # 预测正确 p->p
            else:
                fn += 1  # p->n
            print("cases:", result_cases_distant[index], result_controls_distant[index])
        else:  # 实际结果为controls
            print("control:", result_cases_distant[index], result_controls_distant[index])
            if result_controls_distant[index] > result_cases_distant[index]:
                count_control += 1
                tn += 1  # n->n
            else:
                fp += 1

    f = open(run_path + "/result.csv", 'a', encoding="UTF-8")
    # 写入结果，四列分别是controls中的字符串长度、预测正确的cases比例、预测正确的controls比例、整体预测正确的比例
    result = "%d,%.4f,%.4f,%.4f\n" % (dim, count_case / m, count_control / n, (count_case + count_control) / (m + n))
    f.write(result)
    f.close()
    acc_n, acc_p, accuracy, precision, recall = CA.calculate_apr(tp, fp, fn, tn)
    result = "%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" % ("SM-uncompressed", dim, acc_n, acc_p, accuracy, precision, recall)
    print("%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (dim, acc_n, acc_p, accuracy, precision, recall))
    result_save_path = run_path + "/result_all.csv"
    fp = open(result_save_path, 'a', encoding="UTF-8")
    fp.write(result)
    fp.close()


def calculate_distance_compression(run_path, ratio=0.2):  # 计算距离的评价标准是和样本字符串进行比较(压缩的版本)
    # -------------------------------------------优化top选择比较------------------------------------------
    data_cases_top_tmp, data_controls_top_tmp = get_top_data(run_path)
    data_cases_top = [str_compression(x) for x in data_cases_top_tmp]
    data_controls_top = [str_compression(x) for x in data_controls_top_tmp]  # 进行数据的压缩

    f = open(run_path + "/cases_encoding_str.txt", 'r', encoding="UTF-8")
    data_cases = []
    for line in f:
        data_cases.append(str_compression(line.split(":")[-1]))
    print("cases_encoding_str文件读取完成！")
    f.close()
    data_controls = []
    f = open(run_path + "/controls_encoding_str.txt", 'r', encoding="UTF-8")
    for line in f:
        data_controls.append(str_compression(line.split(":")[-1]))
    print("controls_encoding_str文件读取完成！")
    f.close()

    # --------------------------------------------选择基本的数据---------------------------------------------

    ratio_cases = np.random.randint(0, data_cases.__len__(), int(ratio * data_cases.__len__()))  # 选取20%进行测试
    ratio_control = np.random.randint(0, data_controls.__len__(), int(ratio * data_controls.__len__()))
    print("ratio_cases(count):{}, ratio_controls(count){}".format(ratio_cases.__len__(), ratio_control.__len__()))
    m = ratio_cases.__len__()
    n = ratio_control.__len__()

    Detection_queue = [data_cases[x] for x in ratio_cases] + [data_controls[x] for x in ratio_control]
    result_cases_distant = []
    result_controls_distant = []
    count = 0
    for d in Detection_queue:  # 记录病人的信息
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
    dim_list = [len(x) for x in data_controls] + [len(x) for x in data_cases]
    dim = np.max(np.asarray(dim_list))
    count_case = 0
    count_control = 0
    tp = fp = fn = tn = 0  # 计算得到accuracy,precision,recall 的相关数据
    for index in range(result_controls_distant.__len__()):
        if index < m:
            if result_cases_distant[index] > result_controls_distant[index]:
                count_case += 1
                tp += 1  # 预测正确 p->p
            else:
                fn += 1  # p->n
            print("cases:", result_cases_distant[index], result_controls_distant[index])
        else:
            print("control:", result_cases_distant[index], result_controls_distant[index])
            if result_controls_distant[index] > result_cases_distant[index]:
                count_control += 1
                tn += 1  # n->n
            else:
                fp += 1
    f = open(run_path + "/result.csv", 'a', encoding="UTF-8")
    result = "%d,%.4f,%.4f,%.4f\n" % (dim, count_case / m, count_control / n, (count_case + count_control) / (m + n))
    # print(result)
    f.write(result)
    f.close()
    print("tp=%d, fp=%d, fn=%d, tn=%d" % (tp, fp, fn, tn))
    acc_n, acc_p, accuracy, precision, recall = CA.calculate_apr(tp, fp, fn, tn)
    result = "%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" % ("SM-compressed", dim, acc_n, acc_p, accuracy, precision, recall)
    result_save_path = run_path + "/result_all.csv"
    fp = open(result_save_path, 'a', encoding="UTF-8")
    fp.write(result)
    fp.close()
    return True


def same_length_string(data, k):  # 将字符串转化为相同的长度，将数据对齐为K
    result = []
    for d in data:
        n = d.__len__()
        tem_d = d
        if k > n:
            tem_d = "0" * (k - n) + d
        else:
            tem_d = tem_d[-k:]
        result.append(tem_d)
    return result


# 获取整个样本最好的几个样本,我们认为最大的近似是最优价值的
def top_sample(run_path, ratio=0.2):
    data_cases = []
    names_cases = []
    data_controls = []
    names_controls = []

    path_cases = run_path + "/cases_encoding_str.txt"
    f = open(path_cases, 'r', encoding="UTF-8")
    for line in f:
        data_cases.append(line.split(":")[-1])
        names_cases.append(line.split(":")[0])
    f.close()
    acc_cases = []
    for d in data_cases:
        sum = 0
        for ds in data_cases:
            sum += Levenshtein.jaro(d, ds)
        result = sum / data_cases.__len__()
        acc_cases.append(result)
    result = dict(zip(names_cases, acc_cases))
    result = sorted(result.items(), key=lambda x: -x[-1])
    number = int(data_cases.__len__() * ratio)

    f = open(run_path + "/top_cases.csv", "w", encoding="UTF-8")
    first_line = "name,acc\n"
    f.write(first_line)
    for a in range(number):
        result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
        print(result_tmp)
        f.write(result_tmp)
    f.close()

    path_cases = run_path + "/controls_encoding_str.txt"
    f = open(path_cases, 'r', encoding="UTF-8")
    for line in f:
        data_controls.append(line.split(":")[-1])
        names_controls.append(line.split(":")[0])
    f.close()
    acc_controls = []
    for d in data_controls:
        sum = 0
        for ds in data_controls:
            sum += Levenshtein.jaro(d, ds)
        result = sum / data_controls.__len__()
        acc_controls.append(result)
    result = dict(zip(names_controls, acc_controls))
    result = sorted(result.items(), key=lambda x: -x[-1])
    number = int(data_controls.__len__() * ratio)
    f = open(run_path + "/top_controls.csv", "w", encoding="UTF-8")
    first_line = "name,acc\n"
    f.write(first_line)
    for a in range(number):
        result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
        print(result_tmp)
        f.write(result_tmp)
    f.close()
    return True


# 获取由top_sample计算的结果来获取其数据
def get_top_data(run_path):
    data_cases = []
    data_controls = []
    path_top_cases = run_path + "/top_cases.csv"
    path_top_controls = run_path + "/top_controls.csv"
    # 获取较好样本的名称
    data = pd.read_csv(path_top_cases, sep=',')
    name_top_cases = data["name"].tolist()
    data = pd.read_csv(path_top_controls, sep=',')
    name_top_controls = data["name"].tolist()
    path_str_cases = run_path + "/cases_encoding_str.txt"
    path_str_controls = run_path + "/controls_encoding_str.txt"
    f = open(path_str_cases, 'r', encoding="UTF-8")
    for line in f:
        line_data = line.split(":")
        name = line_data[0]
        if name in name_top_cases:
            data_cases.append(line_data[1])
    f.close()
    f = open(path_str_controls, 'r', encoding="UTF-8")
    for line in f:
        line_data = line.split(":")
        name = line_data[0]
        if name in name_top_controls:
            data_controls.append(line_data[1])
    return data_cases, data_controls


# 主要是为了解决数据的稀疏性问题，指定一个K值，在这个K的基础上进行数据零的压缩，压缩可能会导致长度的不一致
def str_compression(data, k=5):
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


# 另一种简单稀疏编码策略:当一同出现超过k个0的时候手动添加k-1个0
def new_str_compression(data, k=5):
    result = ""
    count = 0
    for d in data:
        if d == "0":
            count += 1
        else:
            if count >= k:
                result += "0" * (k - 1) + d
            else:
                result += "0" * count + d
            count = 0
    if count >= k:
        result += "0" * (k - 1)
    else:
        result += "0" * count
    return result


def string_matching(dataset_path, run_path):
    spindle = SpindleData(dataset_path, run_path)
    spindle.set_bit_coding()
    print("length:%f" % spindle.mean_length)
    print("length:%f" % spindle.max_length)
    spindle.writing_coding_str()
    top_sample(run_path)
    calculate_distance_compression(run_path)


if __name__ == '__main__':
    string_matching("E:\毕业设计\Spindle-master\datasets\mesa_dataset", "E:\毕业设计\Spindle-master\data\mesa")
