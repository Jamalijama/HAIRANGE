import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import optim
import torch
from module_0326 import *
from resnet_18_34 import *

from time import time
import os
import psutil
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.2
dim_max_lst = [768, 768, 768, 512]
# load data
cds = ['pb2', 'pb1', 'pa', 'np']
# cds = ['pb2']
# begin_lst = [60, 60, 60, 40]

begin_lst = [120, 120, 120, 80]

mdl_batch_size = 1150

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, X_valid, y_train, y_valid


for i in range(len(cds)):
    start_time = time()
    # file_name = 'ha_sampled5000_' + cds[i]
    freq_path = '/home/inspur/Downloads/esm-main/WJQ/valid1_fasta/'
    file_name = 'IAV_reassortant_1450to2500_new_4_indp_' + cds[i] + '_emb_esm2_array.npy'
                                       
    label_name = 'IAV_reassortant_1450to2500_new_4_indp_' + cds[i] + '.fasta'
    
    model_path = '/home/inspur/Downloads/HAIRANGE-main/Res/pt/'
    model_name = 'resnet_classification_ha_sampled5000_' + cds[i] +'_ESM2_1150+0.03_resnet34.pt'
    
    inputs = np.load(freq_path + file_name, allow_pickle=True)
    lst_3d = []
    for x0 in inputs:
        # print (x0.shape)
        if x0.shape[0] < dim_max_lst[i]:
            array_zeros = np.zeros((dim_max_lst[i] - x0.shape[0], 1280))
            x = np.concatenate((x0, array_zeros))
        else:
            x = x0
        lst_3d.append(x)
    inputs = np.array(lst_3d)
    print (inputs.shape)

    seq_num = len(inputs)
    
    fr_labels = open (freq_path + label_name, 'r').readlines()
    
    labels0 = []
    for line in fr_labels:
        if line[:1] == '>':
            label = line.split('|')[-1][:-1]
            # print (label)
            labels0.append(label)
    
    labels0 = [int(x) for x in labels0]
    labels = np.array(labels0)
    print (labels.shape)
    
    
    labels = torch.FloatTensor(labels)
    inputs = torch.FloatTensor(inputs) #float64
    
    
    inputs = inputs.view(inputs.size()[0], -1, 128, 128)

    #    batch_size_lst = [1000, 1100, 1150]
    #    lr_lst = [1e-2, 3e-2]

    batch_size_lst = [1150]
    lr_lst = [3e-2]

    model = ResNet34(2, begin_lst[i], dropout)
    model.load_state_dict(torch.load(model_path + model_name))
    model.to(device)
    test_loader = Data.DataLoader(MyDataSet_label(inputs, labels), mdl_batch_size, False, num_workers=0)
    pred, true, prob, valid_acc, valid_f1 = evaluate(model, test_loader)
    
    
    # array_prob = np.array(prob)
    
    # print (cds[i], len(pred), array_prob.shape)
    
    
    df_pred = pd.DataFrame()
    df_pred['predicted'] = pred
    df_pred['prob'] = prob
    df_pred['true'] = labels0
    
    df_pred.to_csv (freq_path + 'df_predicted_' + cds[i] + '.csv')


    # for j in range(n_splits):
    #     train_inputs, valid_inputs, train_labels, valid_labels = get_k_fold_data(n_splits, j, inputs, labels)
    #     train_loader = Data.DataLoader(MyDataSet_label(train_inputs, train_labels), mdl_batch_size, False, num_workers=0)
    #     valid_loader = Data.DataLoader(MyDataSet_label(valid_inputs, valid_labels), mdl_batch_size, False, num_workers=0)
    #     for epoch in range(xx * N_EPOCHS, (xx + 1) * N_EPOCHS):
    #         train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    #         train_loss_lst.append(train_loss)
    #         train_acc_lst.append(train_acc)
    #         print('Epoch: %2d  Train Loss: %.3f  Train Acc: %.2f' % (epoch + 1, train_loss, train_acc * 100))
    
    #         pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
    #         valid_acc_lst.append(valid_acc)
    
    #     pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
    #     pos_prob = Positive_probs(pos_label, pred, prob)
    
    #     trues.append(true)
    #     preds.append(pred)
    #     probs.append(pos_prob)
    #     valid_f1_lst.append(valid_f1)


