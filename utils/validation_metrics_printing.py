import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # DTFDMIL on Paper
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-02-10 130429.703763_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch_BufferSize3072(k=12)(Best)\2024-02-10 130429.704764.txt'
    # DTFDMIL Contrastive
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-03-24 191542.420938_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Bufferv3(w=1.75&e=0.5)Size3072(k=12)\2024-03-24 191542.421938.txt'
    # DTFDMIL Reg
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-04-01 002902.862616_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Bufferv4.5(w=1.25&e=0.5&rw=1.5)Size3072(k=12)\2024-04-01 002902.863616.txt'
    # DTFDMIL Newest
    # r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-04-02 100837.939816_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Buffer+InfoNCE(t=0.5&w=1.75)+Reg(w=1.0)_Size3072(k=12)\2024-04-02 100837.940816.txt'
    # log_file_path = r'E:\WorkGroup\st\Project_ContinualLearning\GraphLearning\weights\TCGA-BRCA\2024-04-02 100837.939816_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Buffer+InfoNCE(t=0.5&w=1.75)+Reg(w=1.0)_Size3072(k=12)\2024-04-02 100837.940816.txt'
    save_root = r'H:\St\SlideGCD_weights\2024-04-14 205037.918821_DTFDMIL_JSDiv(t=1.5&w=1.0)_MFA_FIFO_100epoch(warmup=10)_DSLv2_Buffer+InfoNCE(t=1.25&w=1.75)+Reg(w=1.0)_Size3072(k=12)'
    log_file_path = ''
    for file in os.listdir(save_root):
        if os.path.splitext(file)[-1] == '.txt':
            log_file_path = os.path.join(save_root, file)

    valid_accs, valid_aucs = [], []
    for cur_fold in range(5):
        log_content = []
        with open(log_file_path, 'r') as file:
            presave_idx = 100000
            save_flag = False
            for idx, line in enumerate(file.readlines()):
                if line.find('Training Folder: ' + str(cur_fold) + '.') != -1:
                    presave_idx = idx + 2
                if idx == presave_idx:
                    save_flag = True

                if save_flag:
                    log_content.append(line[:-1])

                # if line.find('Best_ACC_Model: ') != -1:
                if line.find('Best_ACC_Model(Graph): ') != -1:
                    save_flag = False

        for plot_key in ['valid_acc', 'valid_auc']:
            data_graph = []

            max_valid_metric = 0
            for content in log_content:
                # key_graph_idx = content.find(plot_key + ': ')
                key_graph_idx = content.find(plot_key + '(Graph): ')
                if key_graph_idx != -1:
                    # num_begin_idx = key_graph_idx + len(plot_key) + 2
                    num_begin_idx = key_graph_idx + len(plot_key) + 9
                    metric = float(content[num_begin_idx:num_begin_idx + 6])
                    max_valid_metric = np.max([max_valid_metric, metric])
            if plot_key == 'valid_acc':
                valid_accs.append(max_valid_metric)
            else:
                valid_aucs.append(max_valid_metric)
    print("Validation-set ACC: {:.4f}~{:.4f}".format(np.mean(valid_accs) * 100, np.std(valid_accs) * 100))
    print("Validation-set AUC: {:.4f}~{:.4f}".format(np.mean(valid_aucs) * 100, np.std(valid_aucs) * 100))
