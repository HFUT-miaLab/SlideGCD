import argparse
import copy
import os
import sys
import math
import shutil
import datetime
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.spatial.distance import cdist
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, f1_score)
from torch.autograd import Variable

from model.S4MIL import S4Model
from utils import util, metric
import dataset


def train(model, loader, criterion, optimizer, scheduler, args):
    model.train()

    correct, total = 0, 0
    total_loss = 0
    for step, (feat, label) in enumerate(loader):
        optimizer.zero_grad()

        feat, label = feat.cuda(), label.cuda()

        logits = model(feat)['logits']
        loss = criterion(logits, label)

        correct += (torch.argmax(logits.view(feat.shape[0], -1), dim=1) == label).sum().item()
        total += label.shape[0]

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss = total_loss + loss.item()

    return total_loss / total, correct / total


def eval(model, loader, args):
    model.eval()

    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = feat.cuda(), label.cuda()

            bag_prediction = model(feat)['logits']

            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(bag_prediction, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)
    if args.task != 'TCGA-Merged':
        acc, macro_auc, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
    else:
        acc, macro_auc, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)

    return acc, macro_auc


def test(weights, loader, args):
    milnet = S4Model(model_dim=512, state_dim=32, input_dim=args.feats_size, n_classes=len(args.classes)).cuda()
    milnet.load_state_dict(weights)

    milnet.eval()
    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = feat.cuda(), label.cuda()

            bag_prediction = milnet(feat)['logits']

            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(bag_prediction, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)

    return targets, scores


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--task', default='TCGA-NSCLC', type=str,
                        choices=['TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'TCGA-ESCA', 'TCGA-Merged'], help='Dataset folder name')
    parser.add_argument('--classes', default=['LUAD', 'LUSC'], type=list,
                        choices=[['LUAD', 'LUSC'], ['IDC', 'ILC'], ['KIRC', 'KIRP'], ['ESAD', 'ESCC'],
                                 ['LUAD', 'LUSC', 'IDC', 'ILC', 'KIRC', 'KIRP', 'ESAD', 'ESCC']])
    parser.add_argument('--feature_root',
                        default=r'E:\WorkGroup\st\Datasets\features\TCGA_PLIP_features', type=str)
    parser.add_argument('--trainval_csv', default='data/tcga_nsclc_trainval_fold.csv', type=str)
    parser.add_argument('--test_csv', default='data/tcga_nsclc_test.csv', type=str)
    parser.add_argument('--weights_save_path', type=str, default='./weights')

    args = parser.parse_args()

    args.feature_root = os.path.join(args.feature_root, args.task)
    args.weights_save_path = os.path.join(args.weights_save_path, args.task,
                                          datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f'))
    os.makedirs(args.weights_save_path, exist_ok=True)
    sys.stdout = util.Logger(filename=os.path.join(args.weights_save_path,
                                                   datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt'))

    evaluator = metric.MetricEvaluator(classes=args.classes, num_fold=5)
    for fold in range(5):
        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold))
        os.makedirs(args.fold_save_path, exist_ok=True)
        print('Training Folder: {}.\n\tData Loading...'.format(fold))

        # prepare model
        milnet = S4Model(model_dim=512, state_dim=32, input_dim=args.feats_size, n_classes=len(args.classes)).cuda()

        n_parameters = sum(p.numel() for p in milnet.parameters() if p.requires_grad)
        print('Number of requires_grad params (M): %.2f' % (n_parameters / 1.e6))
        n_parameters_full = sum(p.numel() for p in milnet.parameters())
        print('Number of the whole params (M): %.2f' % (n_parameters_full / 1.e6))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr * math.sqrt(args.batch_size),
                                     betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        # loading data
        train_ids, train_labels, train_weights, val_ids, val_labels = dataset.load_trainval(args.trainval_csv, fold=fold)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_labels))
        train_dataset = dataset.OriginDataset(args.feature_root, train_ids, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)
        val_dataset = dataset.OriginDataset(args.feature_root, val_ids, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        best_model_saver = util.BestModelSaver(args.num_epochs, ratio=0)
        for epoch in range(args.num_epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            train_avg_loss, train_acc = train(milnet, train_loader, criterion, optimizer, scheduler, args)

            valid_acc, valid_auc = eval(milnet, val_loader, args)
            best_model_saver.update(valid_acc, valid_auc, epoch)
            print('\t\tEpoch: {} || lr: {:.6f} || train_acc: {:.4f} || train_loss: {:.4f} || valid_acc: {:.4f} || valid_auc: {:.4f}'
                  .format(epoch, current_lr, train_acc, train_avg_loss, valid_acc, valid_auc))

            current_model_weight = copy.deepcopy(milnet.state_dict())
            torch.save(current_model_weight,
                       os.path.join(args.fold_save_path, 'epoch' + str(epoch) + '.pth'))

        shutil.copyfile(
            os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch) + '.pth'),
            os.path.join(args.fold_save_path, 'best_acc.pth'))

        best_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_acc.pth'))

        test_ids, test_labels = dataset.load_test(args.test_csv)
        test_dataset = dataset.OriginDataset(args.feature_root, test_ids, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        targets, scores = test(best_model_weight, test_loader, args)
        if args.task != 'TCGA-Merged':
            acc, macro_auc, micro_auc, f1, _, _, _ = evaluator.update(targets, scores, fold=fold)
        else:
            acc, macro_auc, micro_auc, f1 = evaluator.update(targets, scores, fold=fold)
        evaluator.plot_roc_curve(fold, save_path=os.path.join(args.fold_save_path, 'Plot_ROC_Curve.jpg'))
        evaluator.plot_confusion_matrix(fold, save_path=os.path.join(args.fold_save_path, 'Plot_Confusion_Matrix.jpg'))
        print("\t\tBest_ACC_Model: ACC: {:.4f}, Macro_AUC: {:.4f}, Macro_F1: {:.4f}"
              .format(acc, macro_auc, f1))

    ffv_metrics = evaluator.summary_acc() + evaluator.summary_macro_auc() + evaluator.summary_f1()
    print("Five-Fold-Validation:")
    print("\tBest_ACC_Model: ACC: {:.2f}±{:.2f}, Macro_AUC: {:.2f}±{:.2f}, Macro_F1: {:.2f}±{:.2f}"
          .format(ffv_metrics[0] * 100, ffv_metrics[1] * 100, ffv_metrics[2] * 100,
                  ffv_metrics[3] * 100, ffv_metrics[4] * 100, ffv_metrics[5] * 100))
    evaluator.plot_all_roc_curve(save_path=os.path.join(args.weights_save_path, 'Plot_Whole_ROC_Curve.jpg'))


if __name__ == '__main__':
    main()
