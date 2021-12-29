import keras.backend
from keras.layers import LSTM
from keras import layers
from keras.models import Sequential
from Spindle.preprocessing.preprocessing import SpindleData
from keras.layers import Dense
import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


# ******************************************** Accuracy, Recall, Precision  相关的计算方法*************************
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return _val_precision, _val_recall, _val_f1


def learning_lstm(dataset_path, run_path):  # lstm暂时还是比较适合于文本中，对于有序序暂不合适
    x_train, y_labels, length = data_test(dataset_path, run_path)
    x_train = np.expand_dims(x_train, axis=2)

    model = Sequential()
    model.add(LSTM(48, input_shape=(length, 1)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    print("DIMs=%d" % (x_train[0].shape[0]))
    history = model.fit(x_train, y_labels, epochs=3, batch_size=48, validation_split=0.2)

    metrics = Metrics()
    model.fit(x_train, y_labels, epochs=10, batch_size=48,
              verbose=0, validation_split=0.2,
              callbacks=[metrics])
    precision, recall, f1 = metrics.on_epoch_end(epoch=3)
    print("%.4lf,%.4lf,%.4lf" % (precision, recall, f1))


def learning_gru(dataset_path, run_path):  # gru是其lstm的简化版简化版
    x_train, y_labels, length = data_test(dataset_path, run_path)
    x_train = np.expand_dims(x_train, axis=2)

    model = Sequential()
    model.add(layers.GRU(32, input_shape=(length, 1)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='mae', metrics=['acc'])
    model.summary()

    print("DIMs=%d" % (x_train[0].shape[0]))


def data_test(dataset_path, run_path):
    length = 0  # 每一个系列的长度
    spindle = SpindleData(dataset_path, run_path, step=0.001)
    spindle.set_bit_coding()
    x_train = spindle.coding_q
    y_train = np.asarray(spindle.labels)
    # 将数据打乱
    arr = np.arange(y_train.__len__())
    np.random.shuffle(arr)
    x_train = x_train[arr]
    y_train = y_train[arr]
    length = int(spindle.max_length)
    return x_train, y_train, length


# learning_gru()
learning_lstm("E:\毕业设计\Spindle-master\datasets\mesa_dataset", "E:\毕业设计\Spindle-master\data\mesa")
