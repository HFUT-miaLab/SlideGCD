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
# torch.set_printoptions(profile="full")
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import dsmil
from model.abmil_GraphBuffer_FIFO_MultiscaleFetureAttn import BClassifier
from utils import util, metric
import dataset


def train(model, loader, criterion_main, criterion_dual, optimizer_warmup, scheduler_warmup, optimizer, scheduler, args):
    model.train()

    correct, correct_graph, total = 0, 0, 0
    total_main_loss, total_graph_loss = 0, 0
    for step, (feat, label) in enumerate(loader):
        optimizer.zero_grad()
        optimizer_warmup.zero_grad()

        feat, label = feat.cuda(), label.cuda()
        # print(feat.shape)
        if args.model == 'dsmil':
            pass
            # # refer to dsmil code
            # ins_prediction, bag_prediction, _, _ = model(feat)
            # max_prediction, _ = torch.max(ins_prediction, 0)
            # bag_loss = criterion(bag_prediction.view(args.batch_size, -1), label.view(1, -1))
            # max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))
            # loss = 0.5 * bag_loss + 0.5 * max_loss
        elif args.model == 'abmil':
            logits, logits_graph = model(feat)
            main_loss = criterion_main(logits.view(feat.shape[0], -1), label)
            if args.current_epoch < args.warmup_epochs:
                graph_loss = 0
            else:
                graph_loss = (criterion_main(logits_graph, label)
                              + criterion_dual(torch.log_softmax(logits_graph, dim=1), torch.softmax(logits.detach(), dim=1)))
                # print(criterion_main(logits_graph, label), criterion_dual(torch.log_softmax(logits_graph, dim=1), torch.softmax(logits, dim=1)))

            loss = main_loss + graph_loss
        else:
            raise NotImplementedError

        correct += (torch.argmax(logits.view(feat.shape[0], -1), dim=1) == label).sum().item()
        correct_graph += (torch.argmax(logits_graph.view(feat.shape[0], -1), dim=1) == label).sum().item()
        total += label.shape[0]

        loss.backward()

        if args.current_epoch < args.warmup_epochs:
            optimizer_warmup.step()
            scheduler_warmup.step()
        else:
            optimizer.step()
            scheduler.step()
        total_main_loss += main_loss
        total_graph_loss += graph_loss

    return total_main_loss / total, total_graph_loss / total,  correct / total, correct_graph / total


def eval(model, loader, args):
    model.eval()

    targets, scores, scores_graph = [], [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = feat.cuda(), label.cuda()
            if args.model == 'dsmil':
                # refer to dsmil code
                ins_prediction, bag_prediction, _, _ = model(feat)
            elif args.model == 'abmil':
                logits, logits_graph = model(feat, train=False)
            else:
                raise NotImplementedError
            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist())
            scores_graph.extend(torch.softmax(logits_graph, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores, scores_graph = np.array(targets), np.array(scores), np.array(scores_graph)

    if args.task != 'TCGA-Merged':
        acc, macro_auc, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
        acc_graph, macro_auc_graph, _, _, _, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores_graph)
    else:
        acc, macro_auc, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores)
        acc_graph, macro_auc_graph, _, _ = metric.MetricEvaluator(args.classes).update(targets, scores_graph)

    return acc, macro_auc, acc_graph, macro_auc_graph


def test(weights, loader, args):
    if args.model == 'abmil':
        milnet = BClassifier(args.feats_size, len(args.classes), buffer_size=args.buffer_size).cuda()
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
    targets, scores, scores_graph = [], [], []
    with torch.no_grad():
        for step, (feat, label) in enumerate(loader):
            feat, label = feat.cuda(), label.cuda()
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(feat)
                max_prediction, _ = torch.max(ins_prediction, 0)
            elif args.model == 'abmil':
                    logits, logits_graph = milnet(feat, train=False)
            else:
                raise NotImplementedError

            targets.extend(label.cpu().numpy().tolist())
            scores.extend(torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist())
            scores_graph.extend(torch.softmax(logits_graph, dim=-1).squeeze().cpu().numpy().tolist())
    targets, scores, scores_graph = np.array(targets), np.array(scores), np.array(scores_graph)

    return targets, scores, scores_graph


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of total training epochs')
    parser.add_argument('--buffer_size', default=2048, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='abmil', type=str,
                        choices=['dsmil', 'abmil'], help='MIL model')
    parser.add_argument('--task', default='TCGA-BRCA', type=str,
                        choices=['TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'TCGA-ESCA', 'TCGA-Merged'], help='Dataset folder name')
    parser.add_argument('--classes', default=['IDC', 'ILC'], type=list,
                        choices=[['LUAD', 'LUSC'], ['IDC', 'ILC'], ['KIRC', 'KIRP'], ['ESAD', 'ESCC'],
                                 ['LUAD', 'LUSC', 'IDC', 'ILC', 'KIRC', 'KIRP', 'ESAD', 'ESCC']])
    parser.add_argument('--feature_root', default='data/reduced_features_np8', type=str)
    parser.add_argument('--trainval_csv', default='data/tcga_brca_trainval_fold.csv', type=str)
    parser.add_argument('--test_csv', default='data/tcga_brca_test.csv', type=str)
    parser.add_argument('--weights_save_path', type=str, default='./weights')

    # Reduce Parameters
    parser.add_argument('--num_prototypes', default=8, type=int, help='Number of prototypes per bag')
    args = parser.parse_args()

    args.feature_root = os.path.join(args.feature_root, args.task)
    args.weights_save_path = os.path.join(args.weights_save_path, args.task,
                                          datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f'))
    os.makedirs(args.weights_save_path, exist_ok=True)
    sys.stdout = util.Logger(filename=os.path.join(args.weights_save_path,
                                                   datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt'))

    main_evaluator = metric.MetricEvaluator(classes=args.classes, num_fold=5)
    graph_evaluator = metric.MetricEvaluator(classes=args.classes, num_fold=5)
    for fold in range(5):
        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold))
        os.makedirs(args.fold_save_path, exist_ok=True)
        print('Training Folder: {}.\n\tData Loading...'.format(fold))

        # prepare model
        if args.model == 'abmil':
            milnet = BClassifier(args.feats_size, len(args.classes), buffer_size=args.buffer_size).cuda()
        elif args.model == 'dsmil':
            i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=len(args.classes)).cuda()
            b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=len(args.classes),
                                             dropout_v=0).cuda()
            milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
            state_dict_weights = torch.load('model/dsmil_init.pth')
            milnet.load_state_dict(state_dict_weights, strict=False)
        else:
            raise NotImplementedError

        criterion_main = nn.CrossEntropyLoss()
        criterion_dual = nn.KLDivLoss(reduction='batchmean')
        optimizer_warmup = torch.optim.Adam(milnet.parameters(), lr=args.lr * math.sqrt(args.batch_size),
                                     betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_warmup, args.warmup_epochs, 0.000005)
        optimizer = torch.optim.Adam(milnet.parameters(), lr=0.01 * args.lr * math.sqrt(args.batch_size),
                                     betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs - args.warmup_epochs, 0.000005)

        # loading data
        train_ids, train_labels, train_weights, val_ids, val_labels = dataset.load_trainval(args.trainval_csv, fold=fold)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights))
        train_dataset = dataset.FeatDataset(args.feature_root, train_ids, train_labels)
        # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, pin_memory=True)
        val_dataset = dataset.FeatDataset(args.feature_root, val_ids, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        best_model_saver = util.BestModelSaver(args.num_epochs, ratio=0)
        best_model_saver_graph = util.BestModelSaver(args.num_epochs, ratio=0)
        for epoch in range(args.num_epochs):
            args.current_epoch = epoch
            if args.current_epoch < args.warmup_epochs:
                current_lr = optimizer_warmup.param_groups[0]["lr"]
            else:
                current_lr = optimizer.param_groups[0]["lr"]
            main_loss, graph_loss, train_acc, train_acc_graph = train(milnet, train_loader, criterion_main, criterion_dual, optimizer_warmup, scheduler_warmup, optimizer, scheduler, args)

            valid_acc, valid_auc, valid_acc_graph, valid_auc_graph = eval(milnet, val_loader, args)
            best_model_saver.update(valid_acc, valid_auc, epoch)
            best_model_saver_graph.update(valid_acc_graph, valid_auc_graph, epoch)
            print('\t\tEpoch: {} || lr: {:.6f} || train_acc (Main): {:.4f} || train_acc (Graph): {:.4f} || '
                  'train_loss (Main): {:.4f} || train_loss (Graph): {:.4f} || total_loss: {:.4f} || '
                  'valid_acc: {:.4f} || valid_acc(Graph): {:.4f} || valid_auc: {:.4f} || valid_auc(Graph): {:.4f}'
                  .format(epoch, current_lr, train_acc, train_acc_graph, main_loss, graph_loss, main_loss + graph_loss,
                          valid_acc, valid_acc_graph, valid_auc, valid_auc_graph))

            current_model_weight = copy.deepcopy(milnet.state_dict())
            torch.save(current_model_weight,
                       os.path.join(args.fold_save_path, 'epoch' + str(epoch) + '.pth'))

        shutil.copyfile(
            os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch) + '.pth'),
            os.path.join(args.fold_save_path, 'best_acc.pth'))
        shutil.copyfile(
            os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver_graph.best_valid_acc_epoch) + '.pth'),
            os.path.join(args.fold_save_path, 'best_acc(Graph).pth'))

        test_ids, test_labels = dataset.load_test(args.test_csv)
        test_dataset = dataset.FeatDataset(args.feature_root, test_ids, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        best_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_acc.pth'))
        targets, scores, _ = test(best_model_weight, test_loader, args)
        best_model_weight_graph = torch.load(os.path.join(args.fold_save_path, 'best_acc(Graph).pth'))
        targets, _, scores_graph = test(best_model_weight_graph, test_loader, args)

        if args.task != 'TCGA-Merged':
            acc, macro_auc, micro_auc, f1, _, _, _ = main_evaluator.update(targets, scores, fold=fold)
            acc_graph, macro_auc_graph, micro_auc_graph, f1_graph, _, _, _ = graph_evaluator.update(targets, scores_graph, fold=fold)
        else:
            acc, macro_auc, micro_auc, f1 = main_evaluator.update(targets, scores, fold=fold)
            acc_graph, macro_auc_graph, micro_auc_graph, f1_graph = graph_evaluator.update(targets, scores_graph, fold=fold)

        main_evaluator.plot_roc_curve(fold, save_path=os.path.join(args.fold_save_path, 'Plot_ROC_Curve.jpg'))
        main_evaluator.plot_confusion_matrix(fold, save_path=os.path.join(args.fold_save_path, 'Plot_Confusion_Matrix.jpg'))
        graph_evaluator.plot_roc_curve(fold, save_path=os.path.join(args.fold_save_path, 'Plot_ROC_Curve(Graph_Branch).jpg'))
        graph_evaluator.plot_confusion_matrix(fold, save_path=os.path.join(args.fold_save_path, 'Plot_Confusion_Matrix(Graph_Branch).jpg'))
        print("\t\tBest_ACC_Model: ACC: {:.4f}, Macro_AUC: {:.4f}, Macro_F1: {:.4f}"
              .format(acc, macro_auc, f1))
        print("\t\tBest_ACC_Model(Graph): ACC: {:.4f}, Macro_AUC: {:.4f}, Macro_F1: {:.4f}"
              .format(acc_graph, macro_auc_graph, f1_graph))

    ffv_metrics = main_evaluator.summary_acc() + main_evaluator.summary_macro_auc() + main_evaluator.summary_f1()
    print("Five-Fold-Validation:")
    print("\tBest_ACC_Model: ACC: {:.2f}±{:.2f}, Macro_AUC: {:.2f}±{:.2f}, Macro_F1: {:.2f}±{:.2f}"
          .format(ffv_metrics[0] * 100, ffv_metrics[1] * 100, ffv_metrics[2] * 100,
                  ffv_metrics[3] * 100, ffv_metrics[4] * 100, ffv_metrics[5] * 100))
    main_evaluator.plot_all_roc_curve(save_path=os.path.join(args.weights_save_path, 'Plot_Whole_ROC_Curve.jpg'))

    ffv_metrics_graph = graph_evaluator.summary_acc() + graph_evaluator.summary_macro_auc() + graph_evaluator.summary_f1()
    print("\tBest_ACC_Model(Graph): ACC: {:.2f}±{:.2f}, Macro_AUC: {:.2f}±{:.2f}, Macro_F1: {:.2f}±{:.2f}"
          .format(ffv_metrics_graph[0] * 100, ffv_metrics_graph[1] * 100, ffv_metrics_graph[2] * 100,
                  ffv_metrics_graph[3] * 100, ffv_metrics_graph[4] * 100, ffv_metrics_graph[5] * 100))
    graph_evaluator.plot_all_roc_curve(save_path=os.path.join(args.weights_save_path, 'Plot_Whole_ROC_Curve(Graph).jpg'))


if __name__ == '__main__':
    main()
