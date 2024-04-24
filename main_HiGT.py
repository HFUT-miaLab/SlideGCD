# coding=GB2312
import argparse
import copy
import os
import sys
import math
import shutil
import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_Dataloader
from torch.utils.data import WeightedRandomSampler

from model.HiGT import HiGT
from utils import util, metric
import dataset


def train(model, loader, criterion, optimizer, scheduler, args):
    model.train()

    correct, total = 0, 0
    total_loss = 0
    for step, data in tqdm(enumerate(loader)):
        optimizer.zero_grad()

        data = data.cuda()
        label = data.y

        logits = model(data).unsqueeze(0)
        # print(logits, label)
        loss = criterion(logits, label)

        correct += (torch.argmax(logits, dim=1) == label).sum().item()
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
        for step, data in enumerate(loader):
            data = data.cuda()
            label = data.y

            bag_prediction = model(data).unsqueeze(0)

            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(bag_prediction, dim=-1).cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)

    if args.task != 'TCGA-Merged':
        acc, macro_auc, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
    else:
        acc, macro_auc, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)

    return acc, macro_auc


def test(weights, loader, args):
    milnet = HiGT.HiGT(gcn_in_channels=args.gcn_channel_list[0],
                       gcn_hid_channels=args.gcn_channel_list[1],
                       gcn_out_channels=args.gcn_channel_list[2],
                       gcn_drop_ratio=args.drop_out_ratio,
                       mhit_num=args.mhit_num,
                       fusion_exp_ratio=args.fusion_exp_ratio,
                       out_classes=len(args.classes)).cuda()
    milnet.load_state_dict(weights)

    milnet.eval()
    targets, scores = [], []
    with torch.no_grad():
        for step, data in enumerate(loader):
            data = data.cuda()
            label = data.y

            bag_prediction = milnet(data).unsqueeze(0)

            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(bag_prediction, dim=-1).cpu().numpy().tolist())
    targets, scores = np.array(targets), np.array(scores)

    return targets, scores


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of total training epochs')
    parser.add_argument('--model', default='transmil', type=str,
                        choices=['transmil', 'abmil'], help='MIL model')
    parser.add_argument('--task', default='TCGA-NSCLC', type=str,
                        choices=['TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'TCGA-ESCA', 'TCGA-Merged'],
                        help='Dataset folder name')
    parser.add_argument('--classes', default=['LUAD', 'LUSC'], type=list,
                        choices=[['LUAD', 'LUSC'], ['IDC', 'ILC'], ['KIRC', 'KIRP'], ['ESAD', 'ESCC'],
                                 ['LUAD', 'LUSC', 'IDC', 'ILC', 'KIRC', 'KIRP', 'ESAD', 'ESCC']])
    parser.add_argument('--feature_root',
                        default=r'E:\WorkGroup\st\Datasets\features\HiGT_Graphs', type=str)
    parser.add_argument('--trainval_csv', default='data/tcga_nsclc_trainval_fold(HiGT).csv', type=str)
    parser.add_argument('--test_csv', default='data/tcga_nsclc_test(HiGT).csv', type=str)
    parser.add_argument('--weights_save_path', type=str, default='./weights')

    parser.add_argument("--drop_out_ratio", type=float, default=0.2, help="Drop_out_ratio")
    parser.add_argument("--out_classes", type=int, default=256, help="Model middle dimension")
    parser.add_argument("--gcn_channel_list", type=list, default=[512, 512, 512], help="number of channels in gcn")
    parser.add_argument("--mhit_num", type=int, default=3, help="number of HIViT block")
    parser.add_argument("--fusion_exp_ratio", type=int, default=4, help="expansion ratio of fusion block")
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
        milnet = HiGT.HiGT(gcn_in_channels=args.gcn_channel_list[0],
                           gcn_hid_channels=args.gcn_channel_list[1],
                           gcn_out_channels=args.gcn_channel_list[2],
                           gcn_drop_ratio=args.drop_out_ratio,
                           mhit_num=args.mhit_num,
                           fusion_exp_ratio=args.fusion_exp_ratio,
                           out_classes=len(args.classes)).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr * math.sqrt(args.batch_size),
                                     betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        # loading data
        train_ids, train_labels, train_weights, val_ids, val_labels = dataset.load_trainval(args.trainval_csv,
                                                                                            fold=fold)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_labels))
        train_dataset = dataset.HiGTDataset(args.feature_root, train_ids, train_labels)
        train_loader = pyg_Dataloader(train_dataset, batch_size=1, num_workers=args.num_workers,
                                      sampler=sampler, pin_memory=True)
        val_dataset = dataset.HiGTDataset(args.feature_root, val_ids, val_labels)
        val_loader = pyg_Dataloader(val_dataset, batch_size=1, num_workers=args.num_workers,
                                    shuffle=False, pin_memory=True)

        best_model_saver = util.BestModelSaver(args.num_epochs, ratio=0)
        for epoch in range(args.num_epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            train_avg_loss, train_acc = train(milnet, train_loader, criterion, optimizer, scheduler, args)

            valid_acc, valid_auc = eval(milnet, val_loader, args)
            best_model_saver.update(valid_acc, valid_auc, epoch)
            print(
                '\t\tEpoch: {} || lr: {:.6f} || train_acc: {:.4f} || train_loss: {:.4f} || valid_acc: {:.4f} || valid_auc: {:.4f}'
                .format(epoch, current_lr, train_acc, train_avg_loss, valid_acc, valid_auc))

            current_model_weight = copy.deepcopy(milnet.state_dict())
            torch.save(current_model_weight,
                       os.path.join(args.fold_save_path, 'epoch' + str(epoch) + '.pth'))

        shutil.copyfile(
            os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch) + '.pth'),
            os.path.join(args.fold_save_path, 'best_acc.pth'))

        best_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_acc.pth'))

        test_ids, test_labels = dataset.load_test(args.test_csv)
        test_dataset = dataset.HiGTDataset(args.feature_root, test_ids, test_labels)
        test_loader = pyg_Dataloader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

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
