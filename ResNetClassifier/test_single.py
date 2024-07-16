from random import sample

from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
cds = ['pb2', 'pb1', 'pa', 'np']
begin_lst = [3, 3, 3, 2]
dir = ['', '_trm', '', '_trm']
#file_name = 'after2020_27016_'
file_name = 'IAV_reassortant_indp_'

for i in range(len(cds)):
    freq_path = '../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy'
    label_path = '../Res/npy/array_label_' + file_name + cds[i] + '.npy'
    id_path = '../Res/npy/array_id_' + file_name + cds[i] + '.npy'

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
    loader = Data.DataLoader(MyDataSet_id(inputs, labels, ids), mdl_batch_size, False,
                             num_workers=0)
    # load the model
    model = ResNet34(2, begin_lst[i])
    model.load_state_dict(
        torch.load('../Res/pt/resnet_classification_' + cds[i] + '_1150+0.03_resnet34' + dir[i] + '.pt'))
    model.to(device)
    id, pred, true, prob = test(model, loader)
    fp = 0
    fn = 0
    num = 0
    pos_label = 1
    prob1 = Positive_probs(pos_label, pred, prob)
    label_lst = []
    pred_lst = [pred_.item() for pred_ in pred]
    prob_lst = [prob_.item() for prob_ in prob1]
    for j in range(len(true)):
        if true[j] == 1 and pred[j].item() == 0:
            fp += 1
        elif true[j] == 0 and pred[j].item() == 1:
            fn += 1
        if true[j] != pred[j].item():
            num += 1
            #            print('id:', id[i])
            #            print('pred:', pred[i].item())
            #            print('true:', true[i])
            #            print('prob:', probs[i].item())
            label_lst.append(1)
        else:
            label_lst.append(0)
    print('fp:', fp / seq_num, ' fn:', fn / seq_num)
    print(1 - num / seq_num)

    df = pd.DataFrame()
    df['id'] = id
    df['true'] = true
    df['pred'] = pred_lst
    df['prob'] = prob_lst
    df['test_label'] = label_lst
    df.to_csv('../Res/prediction_result_csv/test_' + file_name + cds[i] + '_1150+0.03_0.2_resnet34' + dir[i] + '.csv',
              index=False)

    trues = []
    preds = []
    probs = []
    valid_f1_lst = []
    
    for k in range(5):
        df_sampled = df.sample(200, random_state=(k + 7))
        trues.append(df_sampled['true'].tolist())
        preds.append(df_sampled['pred'].tolist())
        probs.append(df_sampled['prob'].tolist())
        valid_f1 = f1_score(df_sampled['true'].tolist(), df_sampled['pred'].tolist(), average='macro')
        valid_f1_lst.append(valid_f1)
    PrintROCPRAndCM(trues, preds, probs, pos_label,
                    '../Res/fig/cm_reassortant_' + cds[i] + '_resnet34' + dir[i],
                    '../Res/fig/roc_reassortant_' + cds[i] + '_resnet34' + dir[i] + '.png',
                    '../Res/fig/pr_reassortant_' + cds[i] + '_resnet34' + dir[i] + '.png',
                    '../Res/prediction_result_csv/roc_reassortant_' + cds[i] + '_resnet34' + dir[i] + '.csv',
                    '../Res/prediction_result_csv/pr_reassortant_' + cds[i] + '_resnet34' + dir[i] + '.csv')
    df1 = pd.DataFrame()
    df1['valid_f1'] = valid_f1_lst
    df1.to_csv('../Res/train_result_data/valid_f1_reassortant_' + cds[i] + '_resnet34' + dir[i] + '.csv', index=False)
