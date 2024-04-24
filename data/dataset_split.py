import os
import random
import csv


if __name__ == '__main__':
    neg_slide_list = []
    pos_slide_list = []

    with open('tcga_continual_label.csv', 'r') as file:
        reader = csv.reader(file)
        for row_idx, content in enumerate(reader):
            if content[-1] != '3':
                continue
            if content[-2] == '7':
                neg_slide_list.append(content[:-2])
            elif content[-2] == '8':
                pos_slide_list.append(content[:-2])

    neg_trainvalset = random.sample(neg_slide_list, k=int(0.7 * len(neg_slide_list)))
    pos_trainvalset = random.sample(pos_slide_list, k=int(0.7 * len(pos_slide_list)))

    with open('tcga_esca_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for item in neg_slide_list:
            if item not in neg_trainvalset:
                item.append('0')
                writer.writerow(item)
        for item in pos_slide_list:
            if item not in pos_trainvalset:
                item.append('1')
                writer.writerow(item)

    random.shuffle(neg_trainvalset)
    random.shuffle(pos_trainvalset)
    with open('tcga_esca_trainval_fold.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for idx, content in enumerate(neg_trainvalset):
            content.append('0')
            content.append(idx % 5)
            writer.writerow(content)

        for idx, content in enumerate(pos_trainvalset):
            content.append('1')
            content.append(idx % 5)
            writer.writerow(content)
