# @Time   : 2022/3/15
# @Author : Yiqiao Wang

def calculate_apr(tp, fp, fn, tn):
    '''
        :param tp: the count of true positive
        :param fp: the count of false positive
        :param fn: the count of false negative
        :param tn: the count of true negative

        :return acc_n: the accuracy of negative samples
        :return acc_p: the accuracy of positive samples
        :return accuracy: the accuracy of all samples
        :return precision: the precison of all samples
        :return recall: the recall of all samples
    '''

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc_n = tn / (tn + fp)
    acc_p = tp / (tp + fn)
    return acc_n, acc_p, accuracy, precision, recall