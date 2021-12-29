# 基于纺锤波个数分布的测试
from Spindle.preprocessing.preprocessing import SpindleData
from Spindle.preprocessing.preprocessing import cos
import numpy as np
from Spindle.evaluation.calculate_class_info import CA


# 获得具有典型意义的特征 top=ratio
def top_sample(spindle, ratio=0.2):
    spindle.get_spindle_number_distribution()
    data = spindle.coding_number_distribution_isometic
    data_cases = data[:spindle.cases_n]
    names_cases = spindle.names[:spindle.cases_n]
    data_controls = data[spindle.cases_n:]
    names_controls = spindle.names[spindle.cases_n:]
    # ------------------------病人近似度最高的--------------------
    acc_cases = []
    for d in data_cases:
        sum = 0
        for ds in data_cases:
            sum += cos(d, ds)
        result = sum / data_cases.__len__()
        acc_cases.append(result)
    result = dict(zip(names_cases, acc_cases))
    result = sorted(result.items(), key=lambda x: -x[-1])
    number = int(data_cases.__len__() * ratio)
    print("cases-list")
    for i in range(number):
        print("name:%s,acc:%f" % (result[i][0], result[i][1]))
    cases_top_sample_names = [x[0] for i, x in enumerate(result) if i < number]  # 记录相似度高的病人信息
    # -----------------------正常人的近似度最高---------------------
    acc_controls = []
    for d in data_controls:
        sum = 0
        for ds in data_controls:
            sum += cos(d, ds)
        result = sum / data_controls.__len__()
        acc_controls.append(result)
    result = dict(zip(names_controls, acc_controls))
    result = sorted(result.items(), key=lambda x: -x[-1])
    number = int(data_controls.__len__() * ratio)
    print("controls-list")
    for i in range(number):
        print("name:%s,acc:%f" % (result[i][0], result[i][1]))
    controls_top_sample_names = [x[0] for i, x in enumerate(result) if i < number]
    # --------------------------寻找对应数据---------------------------------------
    cases_top_sample_data = []
    controls_top_sample_data = []
    # 根据名字寻找数据
    for (case_i, case_name) in enumerate(names_cases):
        if case_name in cases_top_sample_names:
            cases_top_sample_data.append(data_cases[case_i])
    for (control_i, control_name) in enumerate(names_controls):
        if control_name in controls_top_sample_names:
            controls_top_sample_data.append(data_controls[control_i])
    return cases_top_sample_data, controls_top_sample_data


# 进行随机的测试
def test_class(spindle, run_path, ratio=0.2):
    cases_top_samples, control_top_samples = top_sample(spindle)
    data_cases = spindle.coding_number_distribution_isometic[:spindle.cases_n]
    data_controls = spindle.coding_number_distribution_isometic[spindle.cases_n:]

    # 测试的序列 比例为ratio=0.2
    m = int(data_cases.__len__() * ratio);
    n = int(data_controls.__len__() * ratio)  # 选取的病人和正常人的个数
    print("cases number:%d, controls number:%d" % (m, n))
    test_cases = np.random.randint(0, data_cases.__len__(), int(data_cases.__len__() * ratio))
    test_control = np.random.randint(0, data_controls.__len__(), int(data_controls.__len__() * ratio))
    # 全部的测试序列
    test_queue = [data_cases[x] for x in test_cases] + [data_controls[x] for x in test_control]
    cases_count = 0
    controls_count = 0
    tp = fp = fn = tn = 0
    for index, test in enumerate(test_queue):
        sum = 0
        for d in cases_top_samples:
            sum += cos(test, d)
        result_cases = sum / cases_top_samples.__len__()
        sum = 0
        for d in control_top_samples:
            sum += cos(test, d)
        result_control = sum / control_top_samples.__len__()
        print("cases:%f controls:%f" % (result_cases, result_control))
        if index < m:  # 实际为cases
            if result_cases > result_control:  # 与cases的相似性更高，即分类为cases
                cases_count += 1
                tp += 1  # 预测正确 p->p
            else:
                fn += 1  # p->n
        else:  # 实际为controls
            if result_control > result_cases:
                controls_count += 1
                tn += 1  # n->n
            else:
                fp += 1
    print("cases_count:%d, control_count:%d" % (cases_count, controls_count))
    acc_n, acc_p, accuracy, precision, recall = CA.calculate_apr(tp, fp, fn, tn)
    result = "%s,%lf,%.4f,%.4f,%.4f,%.4f,%.4f\n" % ("ND", spindle.step, acc_n, acc_p, accuracy, precision, recall)
    print(result)
    result_save_path = run_path + "/result_all.csv"
    fp = open(result_save_path, 'a', encoding="UTF-8")
    fp.write(result)
    fp.close()


def test():
    m = 10
    n = 3
    for i in range(m):
        step = 0.05 * (i + 1)
        for j in range(n):
            spindle = SpindleData("E:\毕业设计\Spindle-master\datasets\mesa_dataset", "E:\毕业设计\Spindle-master\data\mesa", step=step)
            test_class(spindle, "E:\毕业设计\Spindle-master\data\mesa")


if __name__ == '__main__':
    test()
