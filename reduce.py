import argparse
import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.kmeans import KMeans


def reduce(args, slide_list):
    for slide_id in tqdm(slide_list):
        feats = torch.load(os.path.join(args.dataset_root, args.task, slide_id + '.pth'))['feature']
        feats = torch.from_numpy(feats).cuda()

        kmeans = KMeans(n_clusters=args.num_prototypes, max_iter=25).fit(feats)
        centroids = kmeans.cluster_centers_.cpu().numpy()

        torch.save(centroids, os.path.join(args.out_path, args.task, slide_id + '.pth'))
        del feats


if __name__ == '__main__':
    TASK_MAPPING = {'TCGA-NSCLC': 0, 'TCGA-RCC': 1, 'TCGA-BRCA': 2, 'TCGA-ESCA': 3}
    torch.manual_seed(66)  # for reproducibility
    torch.cuda.manual_seed_all(66)  # for reproducibility

    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--task', type=str, default='TCGA-ESCA')
    parser.add_argument('--csv_path', type=str, default='data/tcga_continual_label.csv')
    parser.add_argument('--dataset_root', type=str,
                        default=r'E:\WorkGroup\st\Datasets\features\TCGA_PLIP_features')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--out_path', type=str, default='data/reduced_features')
    args = parser.parse_args()

    args.out_path = args.out_path + '_np' + str(args.num_prototypes)
    os.makedirs(os.path.join(args.out_path, args.task), exist_ok=True)

    slide_list = []
    with open(args.csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row_idx, content in enumerate(reader):
            if row_idx != 0:
                task_id = int(content[-1])
                if task_id == TASK_MAPPING[args.task]:
                    slide_list.append(content[1])
    print("\nLength of dataset:", len(slide_list))
    reduce(args, slide_list)
