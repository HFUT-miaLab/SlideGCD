# coding=GB2312
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # # NodeBufferSIze
    # hyperparameter_index = ['512', '1024', '1536', '2048', '2560', '3072', '3584', '4096']
    # # mean = [92.1120, 90.9560, 92.3980, 91.2500, 91.8280, 92.6840, 91.5360, 92.5420]
    # # std = [1.8624, 2.0769, 0.9799, 1.8337, 1.4616, 1.2445, 2.6520, 2.2351]
    # mean = [93.2440, 93.2280, 93.8980, 93.6340, 93.4240, 94.2320, 92.8700, 94.3100]
    # std = [1.5629, 1.4082, 1.2522, 1.8116, 1.4218, 1.6220, 3.0893, 1.8290]
    # data = {}
    # for i, log in enumerate(hyperparameter_index):
    #     data[hyperparameter_index[i]] = [mean[i], std[i]]
    #
    # # ACC = 90.8160; AUC = 91.7120
    # plt.axhline(91.7120, c='midnightblue', linewidth=2, ls='-.')
    # plt.text(x=-1, y=91.0, s='91.71', rotation=10, c='midnightblue')
    #
    # # 读取数据
    # figdata = pd.DataFrame(data=data)
    # # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    # plt.errorbar(figdata.columns, figdata.loc[0], yerr=figdata.loc[1],
    #              fmt='c--o', ecolor='salmon', capsize=2, mfc='orangered', mec='orangered', ms=5)
    # for i in range(len(hyperparameter_index)):
    #     plt.text(hyperparameter_index[i], data[hyperparameter_index[i]][0] + 0.1, str(round(data[hyperparameter_index[i]][0], 2)),
    #              rotation=20, fontsize=11)
    #
    # # 设置坐标轴标签文本
    # plt.ylim((84, 100))
    # plt.ylabel('Valid_AUC', fontsize=14)
    # plt.grid(True, axis='y')
    # plt.savefig(r'C:\Users\123\Desktop\SlideGCD_Hyperparameters\{}.png'.format('Buffer_Size_AUC'), dpi=400)
    # plt.show()

    # # K
    # hyperparameter_index = ['6', '8', '10', '12', '14', '16']
    # # mean = [92.5400, 91.5380, 93.5440, 92.6840, 92.5400, 92.5380]
    # # std = [0.8752, 1.9400, 1.2074, 1.2445, 1.7374, 1.1886]
    # mean = [94.1180, 93.6840, 93.5520, 94.2320, 93.8580, 94.0720]
    # std = [1.4468, 1.3229, 1.2177, 1.6220, 0.7427, 1.3363]
    # data = {}
    # for i, log in enumerate(hyperparameter_index):
    #     data[hyperparameter_index[i]] = [mean[i], std[i]]
    #
    # # ACC = 90.8160; AUC = 91.7120
    # plt.axhline(91.7120, c='midnightblue', linewidth=2, ls='-.')
    # plt.text(x=-0.75, y=91.0, s='91.71', rotation=10, c='midnightblue')
    #
    # # 读取数据
    # figdata = pd.DataFrame(data=data)
    # # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    # plt.errorbar(figdata.columns, figdata.loc[0], yerr=figdata.loc[1],
    #              fmt='c--o', ecolor='salmon', capsize=2, mfc='orangered', mec='orangered', ms=5)
    # for i in range(len(hyperparameter_index)):
    #     plt.text(hyperparameter_index[i], data[hyperparameter_index[i]][0] + 0.1, str(round(data[hyperparameter_index[i]][0], 2)),
    #              rotation=20, fontsize=11)
    #
    # # 设置坐标轴标签文本
    # plt.ylim((84, 100))
    # plt.ylabel('Valid_AUC', fontsize=14)
    # plt.grid(True, axis='y')
    # plt.savefig(r'C:\Users\123\Desktop\SlideGCD_Hyperparameters\{}.png'.format('K_AUC'), dpi=400)
    # plt.show()

    # # temp_distillation
    # hyperparameter_index = ['0.5', '1.0', '1.5', '2.0']
    # # mean = [92.5420, 91.8260, 92.6840, 92.1100]
    # # std = [0.9770, 1.8344, 1.2445, 1.1943]
    # mean = [93.8200, 92.8460, 94.2320, 93.3020]
    # std = [0.9913, 1.8974, 1.6220, 1.0332]
    # data = {}
    # for i, log in enumerate(hyperparameter_index):
    #     data[hyperparameter_index[i]] = [mean[i], std[i]]
    #
    # # ACC = 90.8160; AUC = 91.7120
    # plt.axhline(91.7120, c='midnightblue', linewidth=2, ls='-.')
    # plt.text(x=-0.43, y=91.0, s='91.71', rotation=10, c='midnightblue')
    #
    # # 读取数据
    # figdata = pd.DataFrame(data=data)
    # # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    # plt.errorbar(figdata.columns, figdata.loc[0], yerr=figdata.loc[1],
    #              fmt='c--o', ecolor='salmon', capsize=2, mfc='orangered', mec='orangered', ms=5)
    # for i in range(len(hyperparameter_index)):
    #     plt.text(hyperparameter_index[i], data[hyperparameter_index[i]][0] + 0.1, str(round(data[hyperparameter_index[i]][0], 2)),
    #              rotation=20, fontsize=11)
    #
    # # 设置坐标轴标签文本
    # plt.ylim((84, 100))
    # plt.ylabel('Valid_AUC', fontsize=14)
    # plt.grid(True, axis='y')
    # plt.savefig(r'C:\Users\123\Desktop\SlideGCD_Hyperparameters\{}.png'.format('temp_Distil_AUC'), dpi=400)
    # plt.show()

    # # buffer_update_loss_weight
    # hyperparameter_index = ['1.0', '1.25', '1.5', '1.75', '2.0']
    # # mean = [91.9660, 92.2560, 92.2540, 92.6840, 92.2560]
    # # std = [1.6611, 1.3807, 1.7783, 1.2445, 2.9764]
    # mean = [93.2980, 93.0200, 93.8280, 94.2320, 93.1100]
    # std = [1.7220, 1.5995, 1.2377, 1.6220, 2.0931]
    # data = {}
    # for i, log in enumerate(hyperparameter_index):
    #     data[hyperparameter_index[i]] = [mean[i], std[i]]
    #
    # # ACC = 90.8160; AUC = 91.7120
    # plt.axhline(91.7120, c='midnightblue', linewidth=2, ls='-.')
    # plt.text(x=-0.57, y=91.0, s='91.71', rotation=10, c='midnightblue')
    #
    # # 读取数据
    # figdata = pd.DataFrame(data=data)
    # # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    # plt.errorbar(figdata.columns, figdata.loc[0], yerr=figdata.loc[1],
    #              fmt='c--o', ecolor='salmon', capsize=2, mfc='orangered', mec='orangered', ms=5)
    # for i in range(len(hyperparameter_index)):
    #     plt.text(hyperparameter_index[i], data[hyperparameter_index[i]][0] + 0.1, str(round(data[hyperparameter_index[i]][0], 2)),
    #              rotation=20, fontsize=11)
    #
    # # 设置坐标轴标签文本
    # plt.ylim((84, 100))
    # plt.ylabel('Valid_AUC', fontsize=14)
    # plt.grid(True, axis='y')
    # plt.savefig(r'C:\Users\123\Desktop\SlideGCD_Hyperparameters\{}.png'.format('Buffer_Update_Loss_Weight_AUC'), dpi=400)
    # plt.show()

    # buffer_update_temperature
    hyperparameter_index = ['0.25', '0.50', '0.75', '1.00', '1.25']
    # mean = [92.1080, 92.6840, 92.2520, 92.5420, 91.9680]
    # std = [1.8716, 1.2445, 1.5360, 0.7425, 1.4523]
    mean = [92.9660, 94.2320, 93.7320, 93.8100, 93.4360]
    std = [1.4347, 1.6220, 1.0682, 1.3994, 1.4123]
    data = {}
    for i, log in enumerate(hyperparameter_index):
        data[hyperparameter_index[i]] = [mean[i], std[i]]

    # ACC = 90.8160; AUC = 91.7120
    plt.axhline(91.7120, c='midnightblue', linewidth=2, ls='-.')
    plt.text(x=-0.57, y=91.0, s='91.71', rotation=10, c='midnightblue')

    # 读取数据
    figdata = pd.DataFrame(data=data)
    # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    plt.errorbar(figdata.columns, figdata.loc[0], yerr=figdata.loc[1],
                 fmt='c--o', ecolor='salmon', capsize=2, mfc='orangered', mec='orangered', ms=5)
    for i in range(len(hyperparameter_index)):
        plt.text(hyperparameter_index[i], data[hyperparameter_index[i]][0] + 0.1, str(round(data[hyperparameter_index[i]][0], 2)),
                 rotation=20, fontsize=11)

    # 设置坐标轴标签文本
    plt.ylim((84, 100))
    plt.ylabel('Valid_AUC', fontsize=14)
    plt.grid(True, axis='y')
    plt.savefig(r'C:\Users\123\Desktop\SlideGCD_Hyperparameters\{}.png'.format('Buffer_Update_Temperature_AUC'), dpi=400)
    plt.show()
