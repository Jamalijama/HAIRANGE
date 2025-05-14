from module_0326 import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# load data
cds = ['pb2', 'pb1', 'pa', 'np']
# begin_lst = [3, 3, 3, 2]
begin_lst = [120, 120, 120, 80]
dir = ['', '_trm', '', '_trm']
# file_name = 'after2020_27016_'

# for i in range(len(cds)):
for i in range(1, 2):
    # freq_path = '../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy'
    # label_path = '../Res/npy/array_label_' + file_name + cds[i] + '.npy'
    # id_path = '../Res/npy/array_id_' + file_name + cds[i] + '.npy'
    
    file_name = 'ha_sampled5000_' + cds[i]
    freq_path = '../Res/model_evaluate/lucaOne/new_AIV_all_8_' + file_name + '_protein_emb_lucaone_array_0503_ordered.npy'
    label_path = '../Res/npy/array_label_' + file_name + '.npy'
    id_path = '../Res/npy/array_id_' + file_name + '.npy'

    inputs = np.load(freq_path, allow_pickle=True)
    seq_num = len(inputs)
    labels = np.load(label_path, allow_pickle=True)
    ids = np.load(id_path, allow_pickle=True)

    inputs = torch.FloatTensor(inputs)
    print(inputs.shape)
    labels = torch.LongTensor(labels)
    # ids = torch.Tensor(ids)

    inputs = inputs.view(inputs.size()[0], -1, 128, 128)
    mdl_batch_size = 1000
    setup_seed(7)
    
    # load the model
    model = ResNet34(2, begin_lst[i])
    model.load_state_dict(
        torch.load('../Res/pt/resnet_classification_' + file_name + '_lucaOne_9_600_1150+0.03_resnet34.pt'))
    # '../Res/pt/resnet_classification_' + file_name + '_lucaOne_' + str(mdl_batch_size) + '+' + str(lr) + '+' + str(N_EPOCHS * n_splits) + '_resnet34.pt')
    model.to(device)
    
    # file_name += '_protein_lucaone_0507_9_600'
    file_name += '_protein_lucaone_0507_9_600_label1'
    
    n_splits = 5
    
    trues = []
    preds = []
    probs = []
    pos_label = 1
    for j in range(n_splits):
        train_inputs, valid_inputs, train_labels, valid_labels = get_k_fold_data(n_splits, j, inputs, labels)
        valid_loader = Data.DataLoader(MyDataSet_label(valid_inputs, valid_labels), mdl_batch_size, False, num_workers=0)
        pred, true, prob = test_no_id(model, valid_loader)
        pos_prob = Positive_probs(pos_label, pred, prob)

        trues.append(true)
        preds.append(pred)
        probs.append(pos_prob)
        # probs.append(prob)
        
    PrintROCPRAndCM(trues, preds, probs, pos_label, '../Res/fig/cm_' + file_name + '_resnet34', 
                    '../Res/fig/roc_' + file_name + '_resnet34.png',
                    '../Res/fig/pr_' + file_name + '_resnet34.png',
                    '../Res/train_result_data/roc_' + file_name + '_resnet34.csv',
                    '../Res/train_result_data/pr_' + file_name + '_resnet34.csv')
    
