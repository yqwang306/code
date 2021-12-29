#!/usr/bin/python
# ----------逻辑回归模型-----------
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import os
import glob


feature = 80  # 特征的数量


def get_distribution_Data(path, dataset_path, key):
    data_r = []
    for im in glob.glob(path + "/*.csv"):
        if dataset_path.find("mesa") != -1:
            data_f = pd.read_csv(im, sep=",")
        elif dataset_path.find("mros") != -1:
            data_f = pd.read_csv(im, skiprows=(0, 1), sep=",")
        else:
            data_f = pd.read_csv(im, sep=",")
        data_f = data_f[key]
        data_r.append(data_f)
    return data_r


def static_spindle_distribution(path, dataset_path):
    step = 0.2  # 通过步长来控制，进行统计信息
    key = "Time_of_night"
    data = get_distribution_Data(path, dataset_path, key)
    result = []
    for tmp_d in data:
        data_count = np.zeros(feature)
        for d in tmp_d:
            data_count[int(d / step)] += 1  # 可以理解为在0-0.2,0.2-0.4,...15.8-16内出现的纺锤波全部认为是出现在最左
        result.append(data_count)  # 以step为间隔，统计每个step出现的纺锤波个数
    length = feature
    print(length)
    x_data = np.full((len(result), length), 0, np.int32)
    # len(result)即所有csv文件的个数，也即样本个数
    for row in range(len(result)):
        length = len(result[row])  # 统一的量化标准（全部转化为相同的维度）
        x_data[row][:length] = result[row]
    return x_data


def deal_info(path):  # 文件的处理，样本的优化             相关的特征选择
    train_data = []
    labels = []
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    for inx, floder in enumerate(cate):
        for im in glob.glob(floder + "/*.csv"):
            label_temp = [0] * 2  # 初始化
            label_temp[inx] = 1  # cases: [1,0]  controls: [0,1]
            labels.append(label_temp)
            print("reading file %s" % im)
    for inx, floder in enumerate(cate):
        data1 = static_spindle_distribution(floder, path)
        train_data.extend(data1)
    return np.asanyarray(train_data, np.float32), np.asarray(labels)


def run(dataset_path):
    data, labels = deal_info(dataset_path)  # 数据处理以及标签，不同模式的数据处理
    print(data.shape[0])
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)  # 随机打乱
    data = data[arr]
    labels = labels[arr]  # 随机数据的处理

    # 训练集和验证集的划分
    ratio = 0.8
    s = np.int(num_example * ratio)
    train_data = data[:s]  # 训练集的准备
    train_labels = labels[:s]

    test_data = data[s:]
    test_labels = labels[s:]  # 测试集的准备

    x = tf.placeholder(tf.float32, shape=[None, feature])
    y = tf.placeholder(tf.float32, shape=[None, 2])

    W = tf.Variable(tf.zeros([feature, 2]))
    b = tf.Variable(tf.zeros([2]))

    actv = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))

    learning_rate = 0.01
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
    accr = tf.reduce_mean(tf.cast(pred, tf.float32))
    init = tf.global_variables_initializer()

    train_epochs = 100
    batch_size = 20
    display_step = 20

    sess = tf.Session()
    sess.run(init)

    for epoch in range(train_epochs):
        num_batch = np.int(train_data.__len__() / batch_size)
        x_start, y_start = 0, 0
        avg_cost = 0
        for i in range(num_batch):
            batch_xs = train_data[x_start:x_start + batch_size]
            batch_ys = train_labels[y_start:y_start + batch_size]
            x_start = x_start + batch_size
            y_start = y_start + batch_size

            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

        # display
        if epoch % display_step == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: test_data, y: test_labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("Epoch: %03d/%03d cost:%.9f train_acc:%.3f test_acc:%.3f" % (
                epoch, train_epochs, avg_cost, train_acc, test_acc))


if __name__ == '__main__':
    run("E:\毕业设计\Spindle-master\datasets\mesa_dataset\\")
