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

from model import abmil, dsmil
from utils import util, metric
import dataset


def train(model, loader, criterion, optimizer, scheduler, args):
    model.train()

    correct, total = 0, 0
    total_loss = 0
    for step, (feat, label) in enumerate(loader):
        optimizer.zero_grad()

        feat, label = feat.cuda(), label.cuda()
        if args.model == 'dsmil':
            # refer to dsmil code
            ins_prediction, bag_prediction, _, _ = model(feat)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(args.batch_size, -1), label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
        elif args.model == 'abmil':
            bag_prediction = model(feat)
            bag_loss = criterion(bag_prediction.view(feat.shape[0], -1), label)
            loss = bag_loss
        else:
            raise NotImplementedError
        correct += (torch.argmax(bag_prediction.view(feat.shape[0], -1), dim=1) == label).sum().item()
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
            if args.model == 'dsmil':
                # refer to dsmil code
                ins_prediction, bag_prediction, _, _ = model(feat)
            elif args.model == 'abmil':
                bag_prediction = model(feat)
            else:
                raise NotImplementedError
            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(bag_prediction, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)
    if len(args.classes) == 2:
        acc, macro_auc, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
    else:
        acc, macro_auc, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)

    return acc, macro_auc


def test(weights, loader, args):
    if args.model == 'abmil':
        milnet = abmil.BClassifier(args.feats_size, len(args.classes)).cuda()
        milnet.load_state_dict(weights)
    elif args.model == 'dsmil':
        i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=len(args.classes)).cuda()
        b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=len(args.classes),
                                         dropout_v=0).cuda()
        milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
        milnet.load_state_dict(weights)
    else:
        raise NotImplementedError

    milnet.eval()
    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = feat.cuda(), label.cuda()
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(feat)
                max_prediction, _ = torch.max(ins_prediction, 0)
            elif args.model == 'abmil':
                bag_prediction = milnet(feat)
            else:
                raise NotImplementedError

            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(bag_prediction, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)

    return targets, scores


def multi_label_roc(labels, predictions, num_classes):
    thresholds, thresholds_optimal, aucs = [], [], []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if len(labels.shape) == 1:
        labels = labels[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='abmil', type=str,
                        choices=['dsmil', 'abmil'], help='MIL model')
    parser.add_argument('--task', default='TCGA-BRCA', type=str,
                        choices=['TCGA-NSCLC', 'TCGA-BRCA'],
                        help='Dataset folder name')
    parser.add_argument('--classes', default=['early-stage', 'late-stage'], type=list,
                        choices=[['I', 'II', 'III', 'IV'], ['early-stage', 'late-stage']])
    parser.add_argument('--feature_root',
                        default=r'E:\WorkGroup\st\Datasets\features\TCGA_PLIP_features', type=str)
    parser.add_argument('--trainval_csv', default='data/tcga_brca_staging_trainval_fold.csv', type=str)
    parser.add_argument('--test_csv', default='data/tcga_brca_staging_test.csv', type=str)
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
        if args.model == 'abmil':
            milnet = abmil.BClassifier(args.feats_size, len(args.classes)).cuda()
        elif args.model == 'dsmil':
            i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=len(args.classes)).cuda()
            b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=len(args.classes),
                                             dropout_v=0).cuda()
            milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
            state_dict_weights = torch.load('model/dsmil_init.pth')
            milnet.load_state_dict(state_dict_weights, strict=False)
        else:
            raise NotImplementedError

        n_parameters = sum(p.numel() for p in milnet.parameters() if p.requires_grad)
        print('Number of requires_grad params (M): %.2f' % (n_parameters / 1.e6))
        n_parameters_full = sum(p.numel() for p in milnet.parameters())
        print('Number of the whole params (M): %.2f' % (n_parameters_full / 1.e6))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr * math.sqrt(args.batch_size),
                                     betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        # loading data
        train_ids, train_labels, train_weights, val_ids, val_labels = dataset.load_trainval_staging(args.trainval_csv,
                                                                                            fold=fold, binary=len(args.classes) == 2)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_labels))
        train_dataset = dataset.OriginDataset(args.feature_root, train_ids, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)
        val_dataset = dataset.OriginDataset(args.feature_root, val_ids, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        best_model_saver = util.BestModelSaver(args.num_epochs, ratio=0.2)
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

        test_ids, test_labels = dataset.load_test_staging(args.test_csv, binary=len(args.classes) == 2)
        test_dataset = dataset.OriginDataset(args.feature_root, test_ids, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        targets, scores = test(best_model_weight, test_loader, args)
        if len(args.classes) == 2:
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
