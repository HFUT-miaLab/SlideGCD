import argparse
import copy
import os
import random
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

from model.DTFD_MIL.network import DTFD_MIL
from utils import util, metric
import dataset


def train(model, loader, criterion, optimizer, args):
    model.train()

    correct, total = 0, 0
    total_loss = 0
    for step, (feat, label) in enumerate(loader):
        optimizer.zero_grad()

        feat = torch.squeeze(feat).cuda()
        label = torch.squeeze(label).cuda()

        bag_pred, sub_preds = model(feat)

        slide_sub_labels = [label for i in range(args.num_group)]
        slide_sub_labels = torch.stack(slide_sub_labels)  ### numGroup
        loss = criterion(sub_preds, slide_sub_labels) + criterion(bag_pred, label.unsqueeze(0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
        loss.backward()
        optimizer.step()

        correct += (torch.argmax(bag_pred, dim=1) == label).sum().item()
        total += args.batch_size
        total_loss = loss.item()

    return total_loss / total, correct / total


def eval(model, loader, args):
    model.eval()

    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat = torch.squeeze(feat).cuda()
            label = torch.squeeze(label).cuda()

            bag_pred, _ = model(feat)

            targets.append(label.cpu().numpy())
            scores.append(torch.softmax(bag_pred, dim=-1).squeeze().cpu().numpy().tolist())

    targets, scores = np.array(targets), np.array(scores)
    if args.task != 'TCGA-Merged':
        acc, macro_auc, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
    else:
        acc, macro_auc, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)

    return acc, macro_auc


def test(weights, loader, args):
    milnet = DTFD_MIL(args.feats_size, len(args.classes), args.total_instance, args.num_group, args.distill_type).cuda()
    milnet.load_state_dict(weights)
    milnet.eval()

    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = torch.squeeze(feat).cuda(), torch.squeeze(label).cuda()

            bag_pred, _ = milnet(feat)

            targets.append(label.cpu().numpy())
            scores.append(torch.softmax(bag_pred, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)

    return targets, scores


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs')
    parser.add_argument('--epoch_step', default=[50], type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='DTFDMIL', type=str,
                        choices=['DTFDMIL'], help='MIL model')
    parser.add_argument('--task', default='TCGA-NSCLC', type=str,
                        choices=['TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'TCGA-ESCA', 'TCGA-Merged'],
                        help='Dataset folder name')
    parser.add_argument('--classes', default=['LUAD', 'LUSC'], type=list,
                        choices=[['LUAD', 'LUSC'], ['IDC', 'ILC'], ['KIRC', 'KIRP'], ['ESAD', 'ESCC'],
                                 ['LUAD', 'LUSC', 'IDC', 'ILC', 'KIRC', 'KIRP', 'ESAD', 'ESCC']])
    parser.add_argument('--feature_root',
                        default=r'E:\WorkGroup\st\Datasets\features\TCGA_PLIP_features', type=str)
    parser.add_argument('--trainval_csv', default='data/tcga_nsclc_trainval_fold.csv', type=str)
    parser.add_argument('--test_csv', default='data/tcga_nsclc_test.csv', type=str)
    parser.add_argument('--weights_save_path', type=str, default='./weights')

    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--num_group', type=int, default=4)
    parser.add_argument('--total_instance', type=int, default=500)
    parser.add_argument('--distill_type', type=str, default='AFS')

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

        # loading data
        train_ids, train_labels, train_weights, val_ids, val_labels = dataset.load_trainval(args.trainval_csv,
                                                                                            fold=fold)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_labels))
        criterion = nn.CrossEntropyLoss()

        # prepare model
        if args.model == 'DTFDMIL':
            milnet = DTFD_MIL(args.feats_size, len(args.classes), args.total_instance, args.num_group, args.distill_type).cuda()
        else:
            raise NotImplementedError

        train_dataset = dataset.OriginDataset(args.feature_root, train_ids, train_labels)
        val_dataset = dataset.OriginDataset(args.feature_root, val_ids, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.epoch_step, gamma=args.lr_decay_ratio)

        best_model_saver = util.BestModelSaver(args.num_epochs, ratio=0)
        for epoch in range(args.num_epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            train_avg_loss, train_acc = train(milnet, loader=train_loader, criterion=criterion, optimizer=optimizer, args=args)
            scheduler.step()

            valid_acc, valid_auc = eval(milnet, loader=val_loader, args=args)
            best_model_saver.update(valid_acc, valid_auc, epoch)
            print(
                '\t\tEpoch: {} || lr: {:.6f} || train_acc: {:.4f} || train_loss: {:.4f} || valid_acc: {:.4f} || valid_auc: {:.4f}'
                .format(epoch, current_lr, train_acc, train_avg_loss, valid_acc, valid_auc))
            torch.save(milnet.state_dict(), os.path.join(args.fold_save_path, 'epoch' + str(epoch) + '.pth'))

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
