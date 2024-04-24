import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # DTFDMIL on Paper
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-02-10 130429.703763_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch_BufferSize3072(k=12)(Best)\2024-02-10 130429.704764.txt'
    # DTFDMIL Contrastive
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-03-24 191542.420938_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Bufferv3(w=1.75&e=0.5)Size3072(k=12)\2024-03-24 191542.421938.txt'
    # DTFDMIL Reg
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-04-01 002902.862616_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Bufferv4.5(w=1.25&e=0.5&rw=1.5)Size3072(k=12)\2024-04-01 002902.863616.txt'
    # DTFDMIL Newest
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-04-02 100837.939816_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Buffer+InfoNCE(t=0.5&w=1.75)+Reg(w=1.0)_Size3072(k=12)\2024-04-02 100837.940816.txt'
    log_file_path = r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-04-02 100837.939816_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Buffer+InfoNCE(t=0.5&w=1.75)+Reg(w=1.0)_Size3072(k=12)\2024-04-02 100837.940816.txt'
    # keys = ['train_acc', 'train_loss', 'valid_acc', 'valid_auc']
    plot_key = 'valid_auc'
    plot_fold = 2

    log_content = []
    with open(log_file_path, 'r') as file:
        presave_idx = 100000
        save_flag = False
        for idx, line in enumerate(file.readlines()):
            if line.find('Training Folder: ' + str(plot_fold) + '.') != -1:
                presave_idx = idx + 2
            if idx == presave_idx:
                save_flag = True

            if save_flag:
                log_content.append(line[:-1])

            if line.find('Best_ACC_Model(Graph): ') != -1:
                save_flag = False

    key = ['train_acc', 'train_loss', 'valid_acc', 'valid_auc']
    assert plot_key in key

    data_main, data_graph = [], []
    if plot_key in ['train_acc', 'train_loss']:
        for content in log_content:
            key_main_idx = content.find(plot_key + ' (Main): ')
            key_graph_idx = content.find(plot_key + ' (Graph): ')
            print(key_main_idx, key_graph_idx)
            if key_main_idx != -1:
                num_begin_idx = key_main_idx + len(plot_key) + 9
                data_main.append(float(content[num_begin_idx:num_begin_idx + 6]))
            if key_graph_idx != -1:
                num_begin_idx = key_graph_idx + len(plot_key) + 10
                data_graph.append(float(content[num_begin_idx:num_begin_idx + 6]))
    else:
        for content in log_content:
            key_main_idx = content.find(plot_key + ': ')
            key_graph_idx = content.find(plot_key + '(Graph): ')
            if key_main_idx != -1:
                num_begin_idx = key_main_idx + len(plot_key) + 2
                data_main.append(float(content[num_begin_idx:num_begin_idx + 6]))
            if key_graph_idx != -1:
                num_begin_idx = key_graph_idx + len(plot_key) + 9
                data_graph.append(float(content[num_begin_idx:num_begin_idx + 6]))

    plt.plot([i + 1 for i in range(len(data_main))], data_main)
    plt.plot([i + 1 for i in range(len(data_graph))], data_graph, c='darkorange')
    plt.legend(['MIL_Branch', 'Graph_Branch'])
    plt.xlabel('Epoch')
    plt.ylabel(plot_key.split('_')[-1])
    plt.title(plot_key)
    plt.show()
