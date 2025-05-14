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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.2

# load data
cds = ['pb2', 'pb1', 'pa', 'np']
# cds = ['pb2']
begin_lst = [120, 120, 120, 80]


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
    file_name = 'ha_sampled5000_' + cds[i]
    freq_path = '../Res/model_evaluate/lucaOne/new_AIV_all_8_' + file_name + '_protein_emb_lucaone_array_0503_ordered.npy'
    label_path = '../Res/npy/array_label_' + file_name + '.npy'

    inputs = np.load(freq_path, allow_pickle=True)
    print (inputs.shape)
    seq_num = len(inputs)
    labels = np.load(label_path, allow_pickle=True)

    inputs = torch.FloatTensor(inputs)
    labels = torch.LongTensor(labels)
    
    inputs = inputs.view(inputs.size()[0], -1, 128, 128)

    #    batch_size_lst = [1000, 1100, 1150]
    #    lr_lst = [1e-2, 3e-2]

    batch_size_lst = [1150]
    lr_lst = [3e-2]

    for mdl_batch_size in batch_size_lst:
        for lr in lr_lst:
            print(mdl_batch_size, lr)

            setup_seed(7)
            
            # Define model and optimizer
            model = ResNet34(2, begin_lst[i], dropout)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate the model
            N_EPOCHS = 80
            n_splits = 5
            file_name += '_lucaOne_7_' + str(N_EPOCHS * n_splits) +'_'
            x = range(0, N_EPOCHS * n_splits)
            skf = StratifiedKFold(n_splits=n_splits)
            xx = 0
            best_valid_acc = float('-inf')
            train_loss_lst = []
            train_acc_lst = []
            valid_acc_lst = []
            valid_f1_lst = []
            trues = []
            preds = []
            probs = []
            pos_label = 1
            for j in range(n_splits):
                train_inputs, valid_inputs, train_labels, valid_labels = get_k_fold_data(n_splits, j, inputs, labels)
                train_loader = Data.DataLoader(MyDataSet_label(train_inputs, train_labels), mdl_batch_size, True, num_workers=0)
                valid_loader = Data.DataLoader(MyDataSet_label(valid_inputs, valid_labels), mdl_batch_size, True, num_workers=0)
                for epoch in range(xx * N_EPOCHS, (xx + 1) * N_EPOCHS):
                    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
                    train_loss_lst.append(train_loss)
                    train_acc_lst.append(train_acc)
                    print('Epoch: %2d  Train Loss: %.3f  Train Acc: %.2f' % (epoch + 1, train_loss, train_acc * 100))

                    pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
                    valid_acc_lst.append(valid_acc)

                pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
                pos_prob = Positive_probs(pos_label, pred, prob)

                trues.append(true)
                preds.append(pred)
                probs.append(pos_prob)
                valid_f1_lst.append(valid_f1)

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(),
                                '../Res/pt/resnet_classification_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.pt')

                xx += 1

            PrintTrain(x, train_loss_lst, valid_acc_lst, '../Res/fig/train_' + file_name + str(mdl_batch_size) +
                        '+' + str(lr) + '_resnet34.png')
            PrintROCPRAndCM(trues, preds, probs, pos_label,
                            '../Res/fig/cm_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34',
                            '../Res/fig/roc_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.png',
                            '../Res/fig/pr_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.png',
                            '../Res/train_result_data/roc_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.csv',
                            '../Res/train_result_data/pr_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.csv')
            df1 = pd.DataFrame()
            df1['train_loss'] = train_loss_lst
            df1['train_acc'] = train_acc_lst
            df1['valid_acc'] = valid_acc_lst
            df1.to_csv('../Res/train_result_data/train_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.csv', index=False)
            
            df1 = pd.DataFrame()
            df1['valid_f1'] = valid_f1_lst
            df1.to_csv('../Res/train_result_data/valid_f1_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.csv', index=False)
            end_time = time()
            elapsed = end_time - start_time
            print(cds[i], N_EPOCHS * n_splits)
            print(elapsed)
        
            info = psutil.virtual_memory()
            print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
            print(u'总内存：', info.total / 1024 / 1024)
            print(u'内存占比：', info.percent)
            print(u'cpu个数：', psutil.cpu_count())
            
            with open('../Res/times/lucaOne_train_result_' + str(N_EPOCHS * n_splits) + '.txt', 'a') as f:
                print(cds[i], file=f)
                print(N_EPOCHS * n_splits, file=f)
                print(elapsed, file=f)
            
                info = psutil.virtual_memory()
                print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, file=f)
                print(u'总内存：', info.total / 1024 / 1024, file=f)
                print(u'内存占比：', info.percent, file=f)
                print(u'cpu个数：', psutil.cpu_count(), file=f)
            f.close()
