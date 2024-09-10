import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def labels_2_one_hot(y, class_num=5):
    one_hot_labels = []
    for i in range(len(y)):
        one_hot = np.zeros(class_num)
        one_hot[y[i]] = 1
        one_hot_labels.append(one_hot.tolist())
    return np.array(one_hot_labels)


def compute_TP_FP_TN_FN(y, pred_labels):
    assert len(y) == len(pred_labels)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1:
            if pred_labels[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred_labels[i] == 0:
                TN += 1
            else:
                FP += 1
    return TP, FP, TN, FN


def compute_specificity(TN, FP):
    return float(TN) / (TN + FP)


def compute_sensitivity(TP, FN):
    return float(TP) / (TP + FN)


def draw_confusion_matrix(y, pred_labels, class_names, save_path=None):
    confusion = confusion_matrix(y, pred_labels)
    _confusion = np.zeros((len(class_names), len(class_names)), dtype=float)
    for pred_index in range(len(confusion)):
        for truth_index in range(len(confusion[pred_index])):
            _confusion[pred_index][truth_index] = confusion[pred_index][truth_index] * 1.0 / np.sum(
                confusion[pred_index]) * 100
    plt.imshow(_confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))

    plt.xticks(indices, class_names)
    plt.yticks(indices, class_names)
    plt.xlabel('Pred')
    plt.ylabel('Truth')
    for pred_index in range(len(confusion)):
        for truth_index in range(len(confusion[pred_index])):
            if _confusion[pred_index][truth_index] >= 70:
                text_color = 'white'
            else:
                text_color = 'black'

            plt.text(truth_index, pred_index, "{:.2f}%".format(
                _confusion[pred_index][truth_index]), ha="center", va="center", color=text_color)
    plt.colorbar()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def draw_binary_roc_curve(y, prob, save_path=None):
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label="AUC (area = {:.3f})".format(roc_auc))
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()
    return roc_auc


def draw_muti_roc_curve(true_labels, pred_probs, sk_macro_auc, class_names=None, save_path=None):
    true_labels = labels_2_one_hot(true_labels, class_num=len(class_names))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(true_labels[0])):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false  positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    colors = ["aqua", "darkorange", "cornflowerblue", "navy", "deeppink", "blue", "purple", "green", "gray"]
    for i, color in zip(range(len(class_names)), colors[:len(class_names)]):
        plt.plot(fpr[i], tpr[i], ls="--", color=color, lw=2, alpha=0.7,
                 label="ROC of {0} (area={1:0.3f})".format(class_names[i], roc_auc[i]))

    plt.plot(fpr["macro"], tpr["macro"], c='r', lw=2, alpha=0.7,
             label="AUC (area = {:.3f})".format(sk_macro_auc))

    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)

    if save_path is None and save_path != "NotShow":
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def macro_auc(y_true, y_score, num_classes):
    if num_classes > 2:
        return roc_auc_score(y_true, y_score, average="macro", multi_class='ovo')
    else:
        y_score = y_score[:, 1]
        return roc_auc_score(y_true, y_score, average="macro")


def micro_auc(y_true, y_score, num_classes):
    if num_classes > 2:
        return roc_auc_score(y_true, y_score, average="micro", multi_class='ovr')
    else:
        y_score = y_score[:, 1]
        return roc_auc_score(y_true, y_score, average="micro")


class MetricEvaluator:
    def __init__(self, classes, num_fold=1):
        self.classes = classes
        self.num_fold = num_fold

        self.targets, self.scores, self.preds = {}, {}, {}

        self._acc = np.zeros(self.num_fold, dtype=np.float32)
        self._precision = np.zeros(self.num_fold, dtype=np.float32)
        self._recall = np.zeros(self.num_fold, dtype=np.float32)
        self._specificity = np.zeros(self.num_fold, dtype=np.float32)
        self._f1 = np.zeros(self.num_fold, dtype=np.float32)
        self._macro_auc = np.zeros(self.num_fold, dtype=np.float32)
        self._micro_auc = np.zeros(self.num_fold, dtype=np.float32)

    def update(self, targets, scores, fold=0):
        preds = np.argmax(scores, axis=1)

        self.targets[fold] = targets
        self.scores[fold] = scores
        self.preds[fold] = preds

        self._acc[fold] = self.calc_acc(fold)
        self._f1[fold] = self.calc_f1(fold)
        self._macro_auc[fold], self._micro_auc[fold] = self.calc_auc(fold)
        if len(self.classes) == 2:
            self._precision[fold], self._recall[fold], self._specificity[fold] = self.calc_p_r(fold)
            return (self._acc[fold], self._macro_auc[fold], self._micro_auc[fold], self._f1[fold],
                    self._precision[fold], self._recall[fold], self._specificity[fold])
        else:
            return self._acc[fold], self._macro_auc[fold], self._micro_auc[fold], self._f1[fold]

    def summary_acc(self):
        return np.mean(self._acc), np.std(self._acc)

    def summary_macro_auc(self):
        return np.mean(self._macro_auc), np.std(self._macro_auc)

    def summary_f1(self):
        return np.mean(self._f1), np.std(self._f1)

    def calc_acc(self, fold):
        return np.sum(self.preds[fold] == self.targets[fold]) / self.preds[fold].shape[0]

    def calc_auc(self, fold):
        return (macro_auc(self.targets[fold], self.scores[fold], len(self.classes)),
                micro_auc(self.targets[fold], self.scores[fold], len(self.classes)))

    def calc_f1(self, fold):
        return f1_score(self.targets[fold], self.preds[fold], average='macro')

    def calc_p_r(self, fold):
        TP, FP, TN, FN = compute_TP_FP_TN_FN(self.targets[fold], self.preds[fold])
        specificity = compute_specificity(TN, FP)
        sensitivity = compute_sensitivity(TP, FN)
        return (precision_score(self.targets[fold], self.preds[fold]),
                recall_score(self.targets[fold], self.preds[fold]),
                specificity)

    def plot_roc_curve(self, fold, save_path=None):
        if len(self.classes) > 2:
            draw_muti_roc_curve(self.targets[fold], self.scores[fold], sk_macro_auc=self._macro_auc[fold],
                                class_names=self.classes, save_path=save_path)
        else:
            draw_binary_roc_curve(self.targets[fold], self.scores[fold][:, 1], save_path=save_path)

    def plot_all_roc_curve(self, save_path=None):
        if len(self.classes) > 2:
            warnings.warn("Plot_all_roc_curve in multi-class situation is not implement", UserWarning)
            return
        else:
            fprs, tprs, roc_aucs = [], [], []
            for i in range(self.num_fold):
                fpr, tpr, _ = roc_curve(self.targets[i], self.scores[i][:, 1])
                fprs.append(fpr)
                tprs.append(tpr)
                roc_aucs.append(auc(fpr, tpr))

            all_fpr = np.unique(np.concatenate([fprs[i] for i in range(self.num_fold)]))
            for i in range(self.num_fold):
                tprs[i] = np.interp(all_fpr, fprs[i], tprs[i])
                tprs[i][0] = 0.0
            mean_tpr = np.mean(tprs, axis=0)
            plt.plot(all_fpr, mean_tpr, c='r', lw=2, alpha=0.7,
                     label='Mean ROC (AUC = {:.3f} ± {:.3f}'.format(np.mean(roc_aucs), np.std(roc_aucs)))

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(all_fpr, tprs_lower, tprs_upper, color='r', alpha=0.2, label='± 1 std. dev.')

            plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
            plt.xlim((-0.01, 1.02))
            plt.ylim((-0.01, 1.02))
            plt.xlabel('False Positive Rate', fontsize=13)
            plt.ylabel('True Positive Rate', fontsize=13)
            plt.title('Receiver operating characteristic example')
            plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    def plot_confusion_matrix(self, fold, save_path=None):
        draw_confusion_matrix(self.targets[fold], self.preds[fold], class_names=self.classes, save_path=save_path)
