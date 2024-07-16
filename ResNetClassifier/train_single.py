import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import optim

from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = 0.2

# load data
cds = ['pb2', 'pb1', 'pa', 'np']
begin_lst = [3, 3, 3, 2]
dir = ['', '_trm', '', '_trm']
file_name = 'before2020_29634_'
file_name1 = 'IAV_reassortant_indp_'


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
    freq_path = '../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy'
    label_path = '../Res/npy/array_label_before2020_29634_' + cds[i] + '.npy'

    inputs = np.load(freq_path, allow_pickle=True)
    seq_num = len(inputs)
    labels = np.load(label_path, allow_pickle=True)

    inputs = torch.FloatTensor(inputs)
    labels = torch.LongTensor(labels)
    inputs = inputs.view(inputs.size()[0], -1, 128, 128)
    
#    freq_path1 = '../Res/npy' + dir[i] + '/array_codon_freq_' + file_name1 + cds[i] + dir[i] + '.npy'
#    label_path1 = '../Res/npy/array_label_' + file_name1 + cds[i] + '.npy'
#
#    inputs1 = np.load(freq_path1, allow_pickle=True)
#    labels1 = np.load(label_path1, allow_pickle=True)
#
#    inputs1 = torch.FloatTensor(inputs1)
#    labels1 = torch.LongTensor(labels1)
#    inputs1 = inputs1.view(inputs1.size()[0], -1, 128, 128)

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
            N_EPOCHS = 10
            n_splits = 5
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
                train_loader = Data.DataLoader(MyDataSet_label(train_inputs, train_labels), mdl_batch_size,
                                               False, num_workers=0)
                valid_loader = Data.DataLoader(MyDataSet_label(valid_inputs, valid_labels), mdl_batch_size,
                                               False, num_workers=0)
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
                               '../Res/pt/resnet_classification_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(
                                   lr) + '_resnet34' + dir[i] + '.pt')

                xx += 1

            PrintTrain(x, train_loss_lst, valid_acc_lst, '../Res/fig/train_' + cds[i] + '_' + str(mdl_batch_size) +
                       '+' + str(lr) + '_resnet34' + dir[i] + '.png')
            PrintROCPRAndCM(trues, preds, probs, pos_label,
                            '../Res/fig/cm_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34' + dir[
                                i],
                            '../Res/fig/roc_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34' + dir[
                                i] + '.png',
                            '../Res/fig/pr_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34' + dir[
                                i] + '.png',
                            '../Res/train_result_data/roc_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(
                                lr) + '_resnet34' + dir[i] + '.csv',
                            '../Res/train_result_data/pr_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(
                                lr) + '_resnet34' + dir[i] + '.csv')
            df1 = pd.DataFrame()
            df1['train_loss'] = train_loss_lst
            df1['valid_acc'] = valid_acc_lst
            df1.to_csv('../Res/train_result_data/train_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(
                lr) + '_resnet34' + dir[i] + '.csv', index=False)
            
            df1 = pd.DataFrame()
            df1['valid_f1'] = valid_f1_lst
            df1.to_csv('../Res/train_result_data/valid_f1_' + cds[i] + '_' + str(mdl_batch_size) + '+' + str(
                lr) + '_resnet34' + dir[i] + '.csv', index=False)
