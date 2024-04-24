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

from model.DTFD_MIL.Attention import Attention_Gated
from model.DTFD_MIL.network import Attention_with_Classifier, DimReduction, Classifier_1fc
from utils import util, metric
import dataset


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


def train(dimReduction, attention, subClassifier, UClassifier, loader, criterion, optimizer0, optimizer1, args):
    dimReduction.train()
    attention.train()
    subClassifier.train()
    UClassifier.train()

    correct, total = 0, 0
    total_loss = 0
    for step, (feat, label) in enumerate(loader):
        feat, label = torch.squeeze(feat).cuda(), torch.squeeze(label).cuda()
        feat_index = list(range(feat.shape[1]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), args.num_group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []
        for tindex in index_chunk_list:
            slide_sub_labels.append(label)
            subFeat = torch.index_select(feat, dim=1, index=torch.LongTensor(tindex).cuda())
            tmidFeat = dimReduction(subFeat)
            tAA = attention(tmidFeat).squeeze(-2)
            tattFeats = torch.einsum('bns,bn->bns', tmidFeat, tAA)  ### n x fs
            af_inst_feat = torch.sum(tattFeats, dim=-2)  ## 1 x fs
            tPredict = subClassifier(af_inst_feat)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            if args.distill_type == 'AFS':
                slide_pseudo_feat.append(torch.unsqueeze(af_inst_feat, dim=1))

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=1)

        ## optimization for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
        loss0 = criterion(slide_sub_preds, slide_sub_labels)
        # print("loss0: ", loss0, slide_sub_preds.shape, slide_sub_labels.shape)
        optimizer0.zero_grad()
        loss0.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), args.grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), args.grad_clipping)
        torch.nn.utils.clip_grad_norm_(subClassifier.parameters(), args.grad_clipping)

        ## optimization for the second tier
        gSlidePred, _ = UClassifier(slide_pseudo_feat)
        loss1 = criterion(gSlidePred, label)
        # print("loss1: ", loss1, gSlidePred.shape, label.shape)
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), args.grad_clipping)
        optimizer0.step()
        optimizer1.step()

        correct += (torch.argmax(gSlidePred, dim=1) == label).sum().item()
        total += args.batch_size
        total_loss += loss0 + loss1

    return total_loss / total, correct / total


def eval(dimReduction, attention, subClassifier, UClassifier, loader, args):
    attention.eval()
    dimReduction.eval()
    subClassifier.eval()
    UClassifier.eval()

    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = torch.squeeze(feat).cuda(), torch.squeeze(label).cuda()
            feat_index = list(range(feat.shape[1]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), args.num_group)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []
            for tindex in index_chunk_list:
                slide_sub_labels.append(label)
                subFeat = torch.index_select(feat, dim=1, index=torch.LongTensor(tindex).cuda())
                tmidFeat = dimReduction(subFeat)
                tAA = attention(tmidFeat).squeeze(-2)
                tattFeats = torch.einsum('bns,bn->bns', tmidFeat, tAA)  ### n x fs
                af_inst_feat = torch.sum(tattFeats, dim=-2)  ## 1 x fs
                tPredict = subClassifier(af_inst_feat)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                if args.distill_type == 'AFS':
                    slide_pseudo_feat.append(torch.unsqueeze(af_inst_feat, dim=1))

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=1)
            bag_prediction, _ = UClassifier(slide_pseudo_feat)

            targets.extend(label.cpu().numpy())
            scores.extend(torch.softmax(bag_prediction, dim=-1).squeeze().cpu().numpy().tolist())

    targets, scores = np.array(targets), np.array(scores)
    if len(args.classes) == 2:
        acc, macro_auc, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
    else:
        acc, macro_auc, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)

    return acc, macro_auc


def test(weights, loader, args):
    attention = Attention_Gated(args.feats_size).cuda()
    dimReduction = DimReduction(args.feats_size, args.feats_size, numLayer_Res=0).cuda()
    subClassifier = Classifier_1fc(args.feats_size, len(args.classes), droprate=0).cuda()
    UClassifier = Attention_with_Classifier(L=args.feats_size, num_cls=len(args.classes), droprate=0).cuda()

    attention.load_state_dict(weights['attention'])
    dimReduction.load_state_dict(weights['dim_reduction'])
    subClassifier.load_state_dict(weights['sub_classifier'])
    UClassifier.load_state_dict(weights['att_classifier'])

    attention.eval()
    dimReduction.eval()
    subClassifier.eval()
    UClassifier.eval()

    targets, scores = [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = torch.squeeze(feat).cuda(), torch.squeeze(label).cuda()
            feat_index = list(range(feat.shape[1]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), args.num_group)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []
            for tindex in index_chunk_list:
                slide_sub_labels.append(label)
                subFeat = torch.index_select(feat, dim=1, index=torch.LongTensor(tindex).cuda())
                tmidFeat = dimReduction(subFeat)
                tAA = attention(tmidFeat).squeeze(-2)
                tattFeats = torch.einsum('bns,bn->bns', tmidFeat, tAA)  ### n x fs
                af_inst_feat = torch.sum(tattFeats, dim=-2)  ## 1 x fs
                tPredict = subClassifier(af_inst_feat)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                if args.distill_type == 'AFS':
                    slide_pseudo_feat.append(torch.unsqueeze(af_inst_feat, dim=1))

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=1)
            bag_prediction, _ = UClassifier(slide_pseudo_feat)

            targets.extend(label.cpu().numpy())
            scores.extend(torch.softmax(bag_prediction, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)

    return targets, scores


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs')
    parser.add_argument('--epoch_step', default=[50], type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='DTFDMIL', type=str,
                        choices=['DTFDMIL'], help='MIL model')
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

    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--num_group', type=int, default=4)
    parser.add_argument('--total_instance', type=int, default=500)
    parser.add_argument('--distill_type', type=str, default='AFS')

    args = parser.parse_args()

    args.feature_root = os.path.join(args.feature_root, args.task)
    if args.model == 'setmil':
        args.feature_root = args.feature_root + '(SETMIL)'
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
        train_ids, train_labels, train_weights, val_ids, val_labels = dataset.load_trainval_staging(args.trainval_csv,
                                                                                            fold=fold, binary=len(args.classes) == 2)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_labels))
        criterion = nn.CrossEntropyLoss()

        # prepare model
        if args.model == 'DTFDMIL':
            dimReduction = DimReduction(args.feats_size, args.feats_size, numLayer_Res=0).cuda()
            attention = Attention_Gated(args.feats_size).cuda()
            subClassifier = Classifier_1fc(args.feats_size, len(args.classes), droprate=0).cuda()
            attCls = Attention_with_Classifier(L=args.feats_size, num_cls=len(args.classes), droprate=0).cuda()

            trainable_parameters = []
            trainable_parameters += list(dimReduction.parameters())
            trainable_parameters += list(attention.parameters())
            trainable_parameters += list(subClassifier.parameters())
        else:
            raise NotImplementedError

        train_dataset = dataset.OriginDataset(args.feature_root, train_ids, train_labels)
        val_dataset = dataset.OriginDataset(args.feature_root, val_ids, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=args.lr * math.sqrt(args.batch_size), weight_decay=args.weight_decay)
        optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=args.lr * math.sqrt(args.batch_size), weight_decay=args.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, args.epoch_step, gamma=args.lr_decay_ratio)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, args.epoch_step, gamma=args.lr_decay_ratio)

        best_model_saver = util.BestModelSaver(args.num_epochs, ratio=0.2)
        for epoch in range(args.num_epochs):
            current_lr = optimizer_adam1.param_groups[0]["lr"]
            train_avg_loss, train_acc = train(dimReduction=dimReduction, attention=attention,
                                              subClassifier=subClassifier, UClassifier=attCls, loader=train_loader,
                                              criterion=criterion, optimizer0=optimizer_adam0,
                                              optimizer1=optimizer_adam1,
                                              args=args)
            scheduler0.step()
            scheduler1.step()

            valid_acc, valid_auc = eval(dimReduction=dimReduction, attention=attention,
                                        subClassifier=subClassifier, UClassifier=attCls, loader=val_loader,
                                        args=args)
            best_model_saver.update(valid_acc, valid_auc, epoch)
            print(
                '\t\tEpoch: {} || lr: {:.6f} || train_acc: {:.4f} || train_loss: {:.4f} || valid_acc: {:.4f} || valid_auc: {:.4f}'
                .format(epoch, current_lr, train_acc, train_avg_loss, valid_acc, valid_auc))

            tsave_dict = {
                'dim_reduction': dimReduction.state_dict(),
                'attention': attention.state_dict(),
                'sub_classifier': subClassifier.state_dict(),
                'att_classifier': attCls.state_dict()
            }
            torch.save(tsave_dict,
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
