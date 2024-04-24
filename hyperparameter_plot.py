import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # K
    data = np.array([[89.81, 89.24, 90.42, 90.39, 89.67], [90.28, 88.60, 89.31, 90.86, 88.84]])
    index = ['6', '8', '10', '12', '14']
    # 绘图
    # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    acc_line = plt.plot(index, data[0, :], 'm*-.', markersize=10)
    auc_line = plt.plot(index, data[1, :], 'bo--', markersize=8)

    # 设置坐标轴标签文本
    plt.ylim((87, 92))
    plt.ylabel('Metric(%)', fontsize=14)
    plt.xlabel('Size of hyperedge', fontsize=14)
    plt.grid(True, axis='y')
    plt.legend(labels=['ACC', 'AUC'], fontsize=14)
    plt.savefig(r'C:\Users\123\Desktop\{}.png'.format('GraphLearning_k'), dpi=400)
    plt.show()

    # # BufferSize
    # data = np.array([[89.53, 89.55, 90.39, 89.81], [90.03, 89.51, 90.86, 90.03]])
    # index = ['1024', '2048', '3072', '4096']
    # # 绘图
    # # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    # acc_line = plt.plot(index, data[0, :], 'm*-.', markersize=10)
    # auc_line = plt.plot(index, data[1, :], 'bo--', markersize=8)
    #
    # # 设置坐标轴标签文本
    # plt.ylim((87, 92))
    # plt.ylabel('Metric(%)', fontsize=14)
    # plt.xlabel('Size of node buffer', fontsize=14)
    # plt.grid(True, axis='y')
    # plt.legend(labels=['ACC', 'AUC'], fontsize=14)
    # plt.savefig(r'C:\Users\123\Desktop\{}.png'.format('GraphLearning_L'), dpi=400)
    # plt.show()

    # # t
    # data = np.array([[89.53, 89.52, 90.39, 89.96], [89.12, 89.44, 90.86, 89.57]])
    # index = ['0.5', '1.0', '1.5', '2.0']
    # # 绘图
    # # 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
    # acc_line = plt.plot(index, data[0, :], 'm*-.', markersize=10)
    # auc_line = plt.plot(index, data[1, :], 'bo--', markersize=8)
    #
    # # 设置坐标轴标签文本
    # plt.ylim((87, 92))
    # plt.ylabel('Metric(%)', fontsize=14)
    # plt.xlabel('Distillation temperature', fontsize=14)
    # plt.grid(True, axis='y')
    # plt.legend(labels=['ACC', 'AUC'], fontsize=14)
    # plt.savefig(r'C:\Users\123\Desktop\{}.png'.format('GraphLearning_t'), dpi=400)
    # plt.show()
