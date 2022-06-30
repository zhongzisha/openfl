import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve, confusion_matrix
from itertools import cycle
import matplotlib.pyplot as plt


def generate_roc_curve(Y, Y_prob, task_name=None, label_names=None, save_filename=None):
    # check epoch 0, using original data
    # skip null value
    # Y:[0, 1, 2] 3
    # Y_pred_prob:
    # [[0.6, 0.3, 0.1],
    # [0.5, 0.4, 0.1],
    # [0.4, 0.2, 0.4]]
    # Y->Y_true_prob: [] N x 3
    # [[1, 0, 0],
    #  [0, 1, 0],
    #  [0, 0, 1]
    # Y.ravale() --> Nx3 --> N1 x 1: [1, 0, 0, 0, 1, 0]
    # micro: Y_true: [1, 0, 0, 0, 1, 0, 0, 0, 1] 9 x 1, Y_pred_prob: [0.6, 0.3, 0.1, ...]

    # resnet18 from ImageNet
    # resnet18 from patch-based MTL
    # resnet18 form DINO (no label, self-supervised)
    # resnet18 form VAE

    # fine-tuning on decoder (multi-task)
    # svs 10000 x (256 x 256)
    # 600 svs, 128 patches -->
    # 1.svs, 128 patches, 32 parts, --> 4 patches / part
    # 2.svs, 256 pathces, 32parts, -->

    Y0 = np.copy(Y)
    Y = label_binarize(Y, classes=[i for i in range(len(label_names))])
    n_classes = Y.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label_counts = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        label_counts.append(len(np.where(Y0 == i)[0]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            # color=color,
            lw=lw,
            label="ROC {0}:{1}({2}) ({3:0.2f})".format(i, label_names[i], label_counts[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(task_name)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_filename)
    plt.close()


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes, task_name="", label_names=None):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.label_names = label_names
        self.task_name = task_name
        self.Y_hat = []
        self.Y = []
        self.Y_prob = []
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y, Y_prob):
        if isinstance(Y_hat, torch.Tensor):
            Y_hat_list = Y_hat.detach().cpu().numpy().tolist()
            Y_list = Y.detach().cpu().numpy().tolist()
            Y_prob = Y_prob.detach().cpu().numpy().tolist()
            self.Y_hat += Y_hat_list
            self.Y += Y_list
            self.Y_prob += Y_prob
            for y_hat, y in zip(Y_hat_list, Y_list):
                self.data[y]["count"] += 1
                self.data[y]["correct"] += (y_hat == y)
        else:
            Y_hat = int(Y_hat)
            Y = int(Y)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = 0
        else:
            acc = float(correct) / count

        return acc, correct, count

    def get_confusion_matrix(self):
        # self.correct_unseen_labels()
        cm = confusion_matrix(y_true=self.Y, y_pred=self.Y_hat)
        return cm

    def get_f1_score(self, average='weighted'):
        # self.correct_unseen_labels()
        score = f1_score(y_true=self.Y, y_pred=self.Y_hat, average=average)
        return score

    def get_auc_score(self, average='weighted'):
        # self.correct_unseen_labels()
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        auc = roc_auc_score(y_true=Y, y_score=Y_prob, average=average, multi_class='ovr')
        return auc

    def get_roc_curve(self, save_filename):
        # self.correct_unseen_labels()
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        generate_roc_curve(Y, Y_prob, task_name=self.task_name,
                           label_names=self.label_names,
                           save_filename=save_filename)

    def correct_unseen_labels(self):
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        Y_labels = np.unique(Y)
        if len(Y_labels) != self.n_classes:
            unseen_labels = set([i for i in range(self.n_classes)]) - set(Y_labels)
            unseen_Y = list(unseen_labels)
            self.Y += unseen_Y
            self.Y_hat += unseen_Y
            self.Y_prob = np.concatenate([self.Y_prob,
                                          label_binarize(unseen_Y, classes=[i for i in range(len(self.label_names))])],
                                         axis=0)
            print('Y', self.Y, np.unique(self.Y), len(self.Y))
            print('Y_prob', self.Y_prob.shape)

    def save_data(self, save_filename=None):
        # self.correct_unseen_labels()
        np.savetxt(save_filename.replace('.txt', '_Y.txt'), X=np.array(self.Y), fmt='%d')
        np.savetxt(save_filename.replace('.txt', '_Y_hat.txt'), X=np.array(self.Y_hat), fmt='%d')
        np.savetxt(save_filename.replace('.txt', '_Y_prob.txt'), X=np.array(self.Y_prob), fmt='%.4f')

