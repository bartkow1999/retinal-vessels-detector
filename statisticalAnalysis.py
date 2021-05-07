from sklearn.metrics import confusion_matrix
import math


def confusion_matrix_data(label, img_true, img_predicted):
    print('\n' + label)
    print('img true shape: ' + str(img_true.shape))
    print('img predicted shape: ' + str(img_predicted.shape))
    TN, FP, FN, TP = confusion_matrix(img_true.flatten(), img_predicted.flatten()).ravel()

    print('TN: ' + str(TN))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TP: ' + str(TP))

    accuracy = float(TP + TN) / float(TP + FP + FN + TN)
    sensitivity = float(TP) / float(TP + FN)
    specificity = float(TN) / float(TN + FP)

    print('accuracy: ' + str(accuracy))
    print('sensitivity: ' + str(sensitivity))
    print('specificity: ' + str(specificity))

    print('geometric mean: ' + str(math.sqrt(specificity * sensitivity)))
