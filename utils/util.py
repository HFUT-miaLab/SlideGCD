import sys
import datetime
import random
import csv

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform


class Logger(object):
    def __init__(self, filename='./logs/' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class BestModelSaver:
    def __init__(self, max_epoch, ratio=0.3, early_stop=10):
        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_acc_epoch = 0

        # Only consider selecting the best model after training beyond this round (begin_epoch)
        self.begin_epoch = int(max_epoch * ratio)
        # Parameter for EarlyStop
        self.early_stop_interval = early_stop
        self.early_stop_count = 0

    def update(self, valid_acc, valid_auc, current_epoch):
        if current_epoch < self.begin_epoch:
            return

        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_valid_auc = valid_auc
            self.best_valid_acc_epoch = current_epoch
            self.early_stop_count = 0
        elif valid_acc == self.best_valid_acc and valid_auc > self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_valid_acc_epoch = current_epoch
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

    def is_early_stop(self):
        return self.early_stop_count >= self.early_stop_interval


class UncertaintyStorage:
    def __init__(self, interval):
        self.interval = interval
        self.logits_storage = []
        self.uncertainty_storage = []

    def update(self, logits):
        self.logits_storage.append(logits)
        self.uncertainty_storage.append(torch.std(logits, dim=1).numpy())

    def compute_uncertainty(self):
        storage = np.array(self.uncertainty_storage)
        print(storage.shape)
        uncertainty = np.mean(storage, axis=0)
        print(uncertainty.shape)
        print(uncertainty)

        self.clear()
        return uncertainty

    def clear(self):
        self.logits_storage = []
        self.uncertainty_storage = []


def fix_random_seeds(seed=None):
    """
    Fix random seeds.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Fix Random Seeds:", seed)


def merge_config_to_args(args, cfg):
    # Data
    args.feature_root = cfg.DATA.FEATURE_ROOT
    args.graphs_root = cfg.DATA.GRAPHS_ROOT
    args.train_valid_csv = cfg.DATA.TRAIN_VALID_CSV
    args.test_csv = cfg.DATA.TEST_CSV

    # Model
    args.arch = cfg.MODEL.ARCH
    args.feat_dim = cfg.MODEL.FEATURE_DIM
    args.num_class = cfg.MODEL.NUM_CLASS
    args.trans_dim = cfg.MODEL.TRANS_DIM
    args.mask_ratio = cfg.MODEL.MASK_RATIO
    args.mask_p = cfg.MODEL.MASK_P
    args.dropout = cfg.MODEL.DROPOUT
    args.loss_weights = cfg.MODEL.LOSS_WEIGHTS

    # TRAIN
    args.batch_size = cfg.TRAIN.BATCH_SIZE
    args.workers = cfg.TRAIN.WORKERS
    args.lr = cfg.TRAIN.LR
    args.weight_decay = cfg.TRAIN.WEIGHT_DECAY
    args.max_epoch = cfg.TRAIN.MAX_EPOCH
    args.show_interval = cfg.TRAIN.SHOW_INTERVAL
    args.eval = cfg.TRAIN.EVAL
    args.weights_save_path = cfg.TRAIN.WEIGHTS_SAVE_PATH


def get_datasplit(csvfile_path, fold):
    train_ids, val_ids, test_ids = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    with open(csvfile_path) as csv_file:
        for line in csv.reader(csv_file):
            slide_id = line[0]
            label = int(line[1])
            trainval_indicate = line[2]

            if trainval_indicate == '':
                test_ids.append(slide_id)
                test_labels.append(label)
            else:
                if int(trainval_indicate) == fold:
                    val_ids.append(slide_id)
                    val_labels.append(label)
                else:
                    train_ids.append(slide_id)
                    train_labels.append(label)

    return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels

def get_datasplit_USTC_EGFR(csvfile_path, fold):
    train_ids, val_ids, test_ids = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    with open(csvfile_path) as csv_file:
        for line in csv.reader(csv_file):
            slide_id = line[0]
            label = int(line[2])
            trainval_indicate = line[3]

            if trainval_indicate == '':
                test_ids.append(slide_id)
                test_labels.append(label)
            else:
                if int(trainval_indicate) == fold:
                    val_ids.append(slide_id)
                    val_labels.append(label)
                else:
                    train_ids.append(slide_id)
                    train_labels.append(label)

    return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels


def euclidean_distances(x):
    return squareform(pdist(x, metric='euclidean'))


def hyperedge_index_concat(hyperedge_index1, hyperedge_index2):
    """
    concate two type of hyperedge_index. e.g. feature-level hyperedge and spatial-level hyperedge.
    :return: concated_hyperedge_index.
    :return: concat_index: where you can split this two kinds of hyperedge.
    """
    hyperedge_index2[1][:] += np.max(hyperedge_index1[1] + 1)
    concated_hyperedge_index = np.hstack((hyperedge_index1, hyperedge_index2))
    return concated_hyperedge_index, hyperedge_index1.shape[1]


def convert_H2index(H):
    """
    convert Adjacency matrix(H) to Sparse Hyperedge_index.
    :param H: Adjacency matrix, shape like (N, E).
    :return: Sparse Hyperedge_index, shape like (2, ?).
    """
    node_list = []
    edge_list = []

    H = np.array(H, dtype=np.float32)  # N * E
    for edge_index in range(H.shape[1]):
        for node_index in range(H.shape[0]):
            if H[node_index][edge_index] != 0:
                node_list.append(node_index)
                edge_list.append(edge_index)
    hyperedge_index = np.vstack((np.array([node_list], dtype=np.int64), np.array([edge_list], dtype=np.int64)))

    return hyperedge_index


def generate_hyperedge_attr(x, hyperedge_index):
    feature_dim = x.shape[1]
    num_hyperedge = np.max(hyperedge_index[1]) + 1

    # Numpy Version
    hyperedge_attr = np.zeros((num_hyperedge, feature_dim), dtype=np.float32)
    for edge_idx in range(num_hyperedge):
        _indexs = hyperedge_index[0][np.argwhere(hyperedge_index[1] == edge_idx).reshape(-1)]

        if _indexs.size == 0:
            continue
        else:
            hyperedge_attr[edge_idx, :] = np.mean(x[_indexs], axis=0, dtype=np.float32)

    return hyperedge_attr


def get_patient_info(csv_file):
    patient_infos = {}

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            patient_id = row[0]
            if row[1] == '男':
                patient_sex = 1
            else:
                patient_sex = 0
            patient_age = int(row[2][:-1])

            patient_infos[patient_id] = {'sex': patient_sex, 'age': patient_age}

    return patient_infos


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
