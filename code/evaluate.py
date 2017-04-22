import numpy as np
from sklearn.metrics import *


def evaluate_1(test_label, test_pred, gettopX=-1):
    precision_list = np.zeros((test_label.shape[1]))
    recall_list = np.zeros((test_label.shape[1]))
    f1_list = np.zeros((test_label.shape[1]))
    accuracy_list = np.zeros((test_label.shape[1]))
    for i in range(test_label.shape[1]):
        cm = confusion_matrix(test_label[:, i],test_pred[:, i])
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        # tn | fp
        # ---|---
        # fn | tp
        precision = tp / float(tp + fp)
        precision_list[i] = precision
        recall = tp / float(tp + fn)
        recall_list[i] = recall
        f1 = 2 * (precision * recall / float(precision + recall))
        f1_list[i] = f1
        accuracy = (tp + tn) / float(tp + tn + fp + fn)
        accuracy_list[i] = accuracy
        print cm
    out = {'prec_mean': np.mean(precision_list),
           'prec_std': np.std(precision_list),
           'recall_mean': np.mean(recall_list),
           'recall_std': np.std(recall_list),
           'acc_mean': np.mean(accuracy_list),
           'acc_std': np.std(accuracy_list),
           'f1_mean': np.mean(f1_list),
           'f1_std': np.std(f1_list)}
    if np.isnan(np.sum(precision_list)):
        out['prec_mean2'] = np.mean(np.nan_to_num(precision_list))
        out['prec_std2'] = np.std(np.nan_to_num(precision_list))
    if np.isnan(np.sum(recall_list)):
        out['recall_mean2'] = np.mean(np.nan_to_num(recall_list))
        out['recall_std2'] = np.std(np.nan_to_num(recall_list))
    if np.isnan(np.sum(f1_list)):
        out['f1_mean2'] = np.mean(np.nan_to_num(f1_list))
        out['f1_std2'] = np.std(np.nan_to_num(f1_list))

    if gettopX > 0:
        idx = np.argsort(np.nan_to_num(f1_list))[-gettopX:]
        out['prec_meantop'] = np.mean(precision_list[idx])
        out['prec_stdtop'] = np.std(precision_list[idx])
        out['recall_meantop'] = np.mean(recall_list[idx])
        out['recall_stdtop'] = np.std(recall_list[idx])
        out['acc_meantop'] = np.mean(accuracy_list[idx])
        out['acc_stdtop'] = np.std(accuracy_list[idx])
        out['f1_meantop'] = np.mean(f1_list[idx])
        out['f1_stdtop'] = np.std(f1_list[idx])
        if np.isnan(np.sum(precision_list[idx])):
            out['prec_meantop2'] = np.mean(np.nan_to_num(precision_list[idx]))
            out['prec_stdtop2'] = np.std(np.nan_to_num(precision_list[idx]))
        if np.isnan(np.sum(recall_list[idx])):
            out['recall_meantop2'] = np.mean(np.nan_to_num(recall_list[idx]))
            out['recall_stdtop2'] = np.std(np.nan_to_num(recall_list[idx]))
        if np.isnan(np.sum(f1_list[idx])):
            out['f1_meantop2'] = np.mean(np.nan_to_num(f1_list[idx]))
            out['f1_stdtop2'] = np.std(np.nan_to_num(f1_list[idx]))

    return out



def evaluate_2(y_true, y_pred, prob = 0.5):
    pred_label = np.copy(y_pred)
    pred_label[pred_label >= prob] = 1
    pred_label[pred_label < prob] = 0

    precision_list = np.zeros((y_true.shape[1]))
    recall_list = np.zeros((y_true.shape[1]))
    f1_list = np.zeros((y_true.shape[1]))
    accuracy_list = np.zeros((y_true.shape[1]))
    auc_list = np.zeros((y_true.shape[1]))
    hamming_loss_list = np.zeros((y_true.shape[1]))
    auc_macro_list =np.zeros((y_true.shape[1]))
    auc_weighted_list = np.zeros((y_true.shape[1]))


    for i in range(y_true.shape[1]):
        sk_precision = precision_score(y_true[:, i], pred_label[:, i])
        sk_recall = recall_score(y_true[:, i], pred_label[:, i])
        sk_f1 = f1_score(y_true[:, i], pred_label[:, i])
        sk_accuracy = accuracy_score(y_true[:, i], pred_label[:, i])

        sk_hamming_loss = hamming_loss(y_true[:, i], pred_label[:, i])
        sk_auc_macro = roc_auc_score(y_true[:, i], y_pred[:, i])
        sk_auc_weighted = roc_auc_score(y_true[:, i], y_pred[:, i], average='weighted')
        precision_list[i] = sk_precision
        recall_list[i] = sk_recall
        f1_list[i] = sk_f1
        accuracy_list[i] = sk_accuracy
        hamming_loss_list[i] = sk_hamming_loss
        auc_macro_list[i] = sk_auc_macro
        auc_weighted_list[i] = sk_auc_weighted


    res = [np.mean(precision_list), np.std(precision_list),
           np.mean(recall_list), np.std(recall_list),
           np.mean(accuracy_list), np.std(accuracy_list),
           np.mean(f1_list), np.std(f1_list),
           np.mean(hamming_loss_list), np.std(hamming_loss_list),
           np.mean(auc_macro_list), np.std(auc_macro_list),
           np.mean(auc_weighted_list), np.std(auc_weighted_list)]

    return res

# get top 5 precision, recall, compare to condensed memory network
def evaluate_3(y_true, y_pred, prob = 0.5):
    pred_label = np.copy(y_pred)
    pred_label[pred_label >= prob] = 1
    pred_label[pred_label < prob] = 0

    precision_list = np.zeros((y_true.shape[1]))
    recall_list = np.zeros((y_true.shape[1]))
    f1_list = np.zeros((y_true.shape[1]))
    accuracy_list = np.zeros((y_true.shape[1]))
    auc_list = np.zeros((y_true.shape[1]))
    hamming_loss_list = np.zeros((y_true.shape[1]))
    auc_macro_list =np.zeros((y_true.shape[1]))
    auc_weighted_list = np.zeros((y_true.shape[1]))


    for i in range(y_true.shape[1]):
        sk_precision = precision_score(y_true[:, i], pred_label[:, i])
        sk_recall = recall_score(y_true[:, i], pred_label[:, i])
        sk_f1 = f1_score(y_true[:, i], pred_label[:, i])
        sk_accuracy = accuracy_score(y_true[:, i], pred_label[:, i])

        sk_hamming_loss = hamming_loss(y_true[:, i], pred_label[:, i])
        sk_auc_macro = roc_auc_score(y_true[:, i], y_pred[:, i])
        sk_auc_weighted = roc_auc_score(y_true[:, i], y_pred[:, i], average='weighted')
        precision_list[i] = sk_precision
        recall_list[i] = sk_recall
        f1_list[i] = sk_f1
        accuracy_list[i] = sk_accuracy
        hamming_loss_list[i] = sk_hamming_loss
        auc_macro_list[i] = sk_auc_macro
        auc_weighted_list[i] = sk_auc_weighted

    precision_list.sort()
    recall_list.sort()
    f1_list.sort()
    precision_list = precision_list[-5:]
    recall_list = recall_list[-5:]
    f1_list = f1_list[-5:]

    res = [np.mean(precision_list), np.std(precision_list),
           np.mean(recall_list), np.std(recall_list),
           np.mean(accuracy_list), np.std(accuracy_list),
           np.mean(f1_list), np.std(f1_list),
           np.mean(hamming_loss_list), np.std(hamming_loss_list),
           np.mean(auc_macro_list), np.std(auc_macro_list),
           np.mean(auc_weighted_list), np.std(auc_weighted_list)]

    return res


def evaluate_3(y_true, y_pred, prob = 0.5):
    pred_label = np.copy(y_pred)
    pred_label[pred_label >= prob] = 1
    pred_label[pred_label < prob] = 0

    precision_list = np.zeros((y_true.shape[1]))
    recall_list = np.zeros((y_true.shape[1]))
    f1_list = np.zeros((y_true.shape[1]))
    accuracy_list = np.zeros((y_true.shape[1]))
    auc_list = np.zeros((y_true.shape[1]))
    hamming_loss_list = np.zeros((y_true.shape[1]))
    auc_macro_list =np.zeros((y_true.shape[1]))
    auc_weighted_list = np.zeros((y_true.shape[1]))


    for i in range(y_true.shape[1]):
        sk_precision = precision_score(y_true[:, i], pred_label[:, i])
        sk_recall = recall_score(y_true[:, i], pred_label[:, i])
        sk_f1 = f1_score(y_true[:, i], pred_label[:, i])
        sk_accuracy = accuracy_score(y_true[:, i], pred_label[:, i])

        sk_hamming_loss = hamming_loss(y_true[:, i], pred_label[:, i])
        sk_auc_macro = roc_auc_score(y_true[:, i], y_pred[:, i])
        sk_auc_weighted = roc_auc_score(y_true[:, i], y_pred[:, i], average='weighted')
        precision_list[i] = sk_precision
        recall_list[i] = sk_recall
        f1_list[i] = sk_f1
        accuracy_list[i] = sk_accuracy
        hamming_loss_list[i] = sk_hamming_loss
        auc_macro_list[i] = sk_auc_macro
        auc_weighted_list[i] = sk_auc_weighted

    idx = np.argsort(np.nan_to_num(f1_list))[-gettopX:]
    precision_list_top5 = precision_list[idx][-5:]
    precision_list_top10 = precision_list[idx][-10:]
    recall_list_top5 = recall_list[idx][-5:]
    recall_list_top10 = recall_list[idx][-10:]
    

    res = [np.mean(precision_list), np.std(precision_list),
           np.mean(recall_list), np.std(recall_list),
           np.mean(accuracy_list), np.std(accuracy_list),
           np.mean(f1_list), np.std(f1_list),
           np.mean(hamming_loss_list), np.std(hamming_loss_list),
           np.mean(auc_macro_list), np.std(auc_macro_list),
           np.mean(auc_weighted_list), np.std(auc_weighted_list),
           np.mean(f1_list_top5), np.std(f1_list_top5),
           np.mean(f1_list_top10), np.std(f1_list_top10),
           np.mean(precision_list_top5), np.std(precision_list_top5),
           np.mean(precision_list_top10), np.std(precision_list_top10),
           np.mean(recall_list_top5), np.std(recall_list_top5),
           np.mean(recall_list_top10), np.std(recall_list_top10)
           ]

    return res


def evaluate_4(y_true, y_pred):
    res_list = np.zeros((10, 6))
    for ii in range(1, 11):
        prob = ii / 10.0
	pred_label = np.copy(y_pred)
        pred_label[pred_label >= prob] = 1
        pred_label[pred_label < prob] = 0

        precision_list = np.zeros((y_true.shape[1]))
        recall_list = np.zeros((y_true.shape[1]))
        f1_list = np.zeros((y_true.shape[1]))
        accuracy_list = np.zeros((y_true.shape[1]))
        auc_list = np.zeros((y_true.shape[1]))
        hamming_loss_list = np.zeros((y_true.shape[1]))
        auc_macro_list = np.zeros((y_true.shape[1]))
        auc_weighted_list = np.zeros((y_true.shape[1]))

        for i in range(y_true.shape[1]):
            sk_precision = precision_score(y_true[:, i], pred_label[:, i])
            sk_recall = recall_score(y_true[:, i], pred_label[:, i])
            sk_f1 = f1_score(y_true[:, i], pred_label[:, i])
            sk_accuracy = accuracy_score(y_true[:, i], pred_label[:, i])

            sk_hamming_loss = hamming_loss(y_true[:, i], pred_label[:, i])
            sk_auc_macro = roc_auc_score(y_true[:, i], y_pred[:, i])
            sk_auc_weighted = roc_auc_score(y_true[:, i], y_pred[:, i], average='weighted')
            precision_list[i] = sk_precision
            recall_list[i] = sk_recall
            f1_list[i] = sk_f1
            accuracy_list[i] = sk_accuracy
            hamming_loss_list[i] = sk_hamming_loss
            auc_macro_list[i] = sk_auc_macro
            auc_weighted_list[i] = sk_auc_weighted

        
        idx = np.argsort(np.nan_to_num(f1_list))[-gettopX:]
        precision_list_top5 = precision_list[idx][-5:]
        precision_list_top10 = precision_list[idx][-10:]
        recall_list_top5 = recall_list[idx][-5:]
        recall_list_top10 = recall_list[idx][-10:]

        res_list[ii-1, :] = [np.mean(precision_list),
               np.mean(recall_list),
               np.mean(precision_list_top5),
               np.mean(recall_list_top5),
               np.mean(precision_list_top10),
               np.mean(recall_list_top5)
               ]



    return res_list
