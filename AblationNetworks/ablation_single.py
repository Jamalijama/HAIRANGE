from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cds = ['pb2', 'pb1', 'pa', 'np']
begin_lst = [3, 3, 3, 2]
seqlen_opt_lst = [760, 758, 717, 499]
dir = ['', '_trm', '', '_trm']
file_name = 'before2020_29634_'

# load data
for i in range(len(cds)):
    freq_path = '../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy'
    label_path = '../Res/npy/array_label_' + file_name + cds[i] + '.npy'
    id_path = '../Res/npy/array_id_' + file_name + cds[i] + '.npy'
    print(freq_path, label_path, id_path)

    all_inputs = np.load(freq_path, allow_pickle=True)
    seq_num = len(all_inputs)
    all_inputs = torch.FloatTensor(all_inputs)
    length = all_inputs.shape[1]
    labels = np.load(label_path, allow_pickle=True)
    ids = np.load(id_path, allow_pickle=True)
    labels = torch.LongTensor(labels)

    window = 80
    seqlen_opt = seqlen_opt_lst[i]
    reverse_acc_lst = []

    for j in range(seqlen_opt):
        inputs = all_inputs.clone()
        if j + window <= seqlen_opt:
            inputs[:, j:j + window, :] = 0
        else:
            inputs[:, j:seqlen_opt, :] = 0
            inputs[:, :(j + window - seqlen_opt), :] = 0

        inputs = inputs.view(inputs.size()[0], -1, 128, 128)

        mdl_batch_size = 1000

        setup_seed(7)
        loader = Data.DataLoader(MyDataSet_id(inputs, labels, ids), mdl_batch_size, False, num_workers=0)
        # load the model
        model = ResNet34(2, begin_lst[i])
        #        model.load_state_dict(
        #            torch.load('../Res1/pt/resnet_classification_' + cds[i] + '_1150+0.03_0.2_resnet34' + dir[i] + '.pt'))
        model.load_state_dict(
            torch.load('../Res/pt/resnet_classification_' + cds[i] + '_1150+0.03_resnet34' + dir[i] + '.pt'))
        model.to(device)
        #        print('./pt/resnet_classification_' + cds[i] + '_1150+0.03_0.2_resnet34' + dir[i] + '.pt')

        id, pred, true, probs = test(model, loader)
        num = 0
        for k in range(len(true)):
            if true[k] != pred[k].item():
                num += 1
        print(j, num / seq_num)
        reverse_acc_lst.append(num / seq_num)

    # print(len(reverse_acc_lst))
    # print(reverse_acc_lst)

    vecter_importance_lst = [0 for _ in range(seqlen_opt)]
    for k in range(seqlen_opt):
        if k >= window - 1:
            for j in range(k - window + 1, k + 1):
                vecter_importance_lst[k] += reverse_acc_lst[j]
        else:
            for j in range(k + 1):
                vecter_importance_lst[k] += reverse_acc_lst[j]
            for j in range(window - k - 1):
                vecter_importance_lst[k] += reverse_acc_lst[seqlen_opt - j - 1]
        vecter_importance_lst[k] /= window

    print(vecter_importance_lst)

    coden_importance_lst = [0 for _ in range(seqlen_opt)]
    for k in range(seqlen_opt):
        if k >= 64 - 1:
            for j in range(k - 64 + 1, k + 1):
                coden_importance_lst[k] += vecter_importance_lst[j]
        else:
            for j in range(k + 1):
                coden_importance_lst[k] += vecter_importance_lst[j]
            for j in range(64 - k - 1):
                coden_importance_lst[k] += vecter_importance_lst[seqlen_opt - j - 1]

        if k + 64 <= seqlen_opt:
            for j in range(k, k + 64):
                coden_importance_lst[k] += vecter_importance_lst[j]
        else:
            for j in range(k, seqlen_opt):
                coden_importance_lst[k] += vecter_importance_lst[j]
            for j in range(k + 64 - seqlen_opt):
                coden_importance_lst[k] += vecter_importance_lst[j]
        coden_importance_lst[k] /= 128

    print(coden_importance_lst)

    df = pd.DataFrame()
    df['resnet'] = coden_importance_lst
    df.to_csv('../Res/ablation_result/' + cds[i] + '_' + str(window) + '_1150+0.03_0.2_resnet34' + dir[i] + '.csv',
              index=False)
