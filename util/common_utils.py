from sklearn import metrics

class IndicatorCalculation():  # 包含二分类中各种指标
    '''
    tp, fp
    fn, tn

    '''

    def __init__(self, prediction=None, ground_truth=None):
        if prediction is not None and ground_truth is not None:
            self.prediction = prediction  # [0, 1, 0, 1, 1, 0]
            self.ground_truth = ground_truth  # [0, 1, 0, 0, 1 ]

    @staticmethod
    def __division_detection(number):  # division detection if divisor is zero, the result is zero
        return 0 if number == 0 else number

    def __tp(self):
        TP = 0
        for i in range(len(self.prediction)):
            TP += 1 if self.prediction[i] == 1 and self.ground_truth[i] == 1 else 0
        return TP

    def __fp(self):
        FP = 0
        for i in range(len(self.prediction)):
            FP += 1 if self.prediction[i] == 1 and self.ground_truth[i] == 0 else 0
        return FP

    def __fn(self):
        FN = 0
        for i in range(len(self.prediction)):
            FN += 1 if self.prediction[i] == 0 and self.ground_truth[i] == 1 else 0
        return FN

    def __tn(self):
        TN = 0
        for i in range(len(self.prediction)):
            TN += 1 if self.prediction[i] == 0 and self.ground_truth[i] == 0 else 0
        return TN

    def set_values(self, prediction, ground_truth):
        self.prediction = prediction
        self.ground_truth = ground_truth

    def get_accuracy(self):
        return (self.__tp() + self.__tn()) / (self.__tn() + self.__tp() + self.__fn() + self.__fp())

    def get_recall(self):
        divisor = self.__division_detection(self.__tp() + self.__fn())
        if divisor == 0:
            return 0
        else:
            return self.__tp() / divisor

    def get_precision(self):
        divisor = self.__division_detection(self.__tp() + self.__fp())
        if divisor == 0:
            return 0
        else:
            return self.__tp() / divisor

    def get_f1score(self):
        if (self.get_recall() is None) or (self.get_precision() is None) or (
                (self.get_recall() + self.get_precision()) == 0):
            return 0
        else:
            return (2 * self.__tp()) / (2 * self.__tp() + self.__fn() + self.__fp())

    def get_auc(self, y_pre=None, y_real=None):
        # if type(self.prediction) is not np.ndarray:
        #     self.prediction = np.asarray(self.prediction)
        #     self.ground_truth = np.asarray(self.ground_truth)
        if y_real is  None and y_pre is  None:
            y_predict = self.prediction.cpu()
            y_real = self.ground_truth.cpu()
        else:
            y_predict = y_pre.cpu()
            y_real = y_real.cpu()
        auc_score = metrics.roc_auc_score(y_real, y_predict)
        return auc_score
