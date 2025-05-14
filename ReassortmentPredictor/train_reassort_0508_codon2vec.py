import numpy as np
from sklearn.model_selection import train_test_split
from torch import optim

from module_0326 import *
from resnet_18_34 import *


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

file_name = 'ha_sampled5000_'
inputs = np.load('../Res/npy_reassort/array_codon_freq_' + file_name + '0208_2816.npy', allow_pickle=True)
labels = np.load('../Res/npy/array_label_' + file_name + 'pb2.npy', allow_pickle=True)

inputs = torch.FloatTensor(inputs)
labels = torch.LongTensor(labels)
print(inputs.shape, labels.shape)


mdl_batch_size = 500
lr = 3e-2
inputs = inputs.view(inputs.size()[0], -1, 128, 128)
setup_seed(17)

model = ResNet34(2, 11)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Train and evaluate the model
N_EPOCHS = 6
n_splits = 5
x = range(0, N_EPOCHS * n_splits)
xx = 0

file_name += 'Codon2Vec_0508_seed17_'

best_valid_acc = float('-inf')
train_loss_lst = []
train_acc_lst = []
valid_acc_lst = []
valid_f1_lst = []
pos_label = 1
trues = []
preds = []
probs = []
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
        torch.save(model.state_dict(), '../Res/pt/resnet_classification_' + str(mdl_batch_size) + '+' + str(
                      lr) + '_' + str(epoch) + '_resnet34_reassort_' + file_name + '2816.pt')

    pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
    pos_prob = Positive_probs(pos_label, pred, prob)
    trues.append(true)
    preds.append(pred)
    probs.append(pos_prob)
    valid_f1_lst.append(valid_f1)
    
    # if valid_acc > best_valid_acc:
    #                 best_valid_acc = valid_acc
    #                 torch.save(model.state_dict(),
    #                            '../Res/pt/resnet_classification_' + file_name + str(mdl_batch_size) + '+' + str(lr) + '_resnet34.pt')

    xx += 1

PrintTrain(x, train_loss_lst, valid_acc_lst,
           '../Res/fig/train_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34_reassort_' + file_name + '2816.png')
PrintROCPRAndCM(trues, preds, probs, pos_label,
                    '../Res/fig/cm_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34_reassort_' + file_name + '2816',
                    '../Res/fig/roc_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34_reassort_' + file_name + '2816.png',
                    '../Res/fig/pr_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34_reassort_' + file_name + '2816.png',
                    '../Res/train_result_data/roc_' + str(mdl_batch_size) + '+' + str(lr) + '_' + str(
                        xx) + '_resnet34_reassort_' + file_name + '2816.csv',
                    '../Res/train_result_data/pr_' + str(mdl_batch_size) + '+' + str(lr) + '_' + str(
                        xx) + '_resnet34_reassort_' + file_name + '2816.csv')
df1 = pd.DataFrame()
df1['train_loss'] = train_loss_lst
df1['valid_acc'] = valid_acc_lst
df1.to_csv('../Res/train_result_data/train_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34_reassort_' + file_name + '2816.csv',
           index=False)

df1 = pd.DataFrame()
df1['valid_f1'] = valid_f1_lst
df1.to_csv('../Res/train_result_data/valid_f1_' + str(mdl_batch_size) + '+' + str(lr) + '_resnet34_reassort_' + file_name + '2816.csv', 
           index=False)

