from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
cds = ['pb2', 'pb1', 'pa', 'np']
# begin_lst = [3, 3, 3, 2]
begin_lst = [120, 120, 120, 80]
dir = ['', '', '_trm', '_trm']
# file_name = 'after2020_27016_'

for i in range(len(cds)):
# for i in range(2, 3):
    # freq_path = '../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy'
    # label_path = '../Res/npy/array_label_' + file_name + cds[i] + '.npy'
    # id_path = '../Res/npy/array_id_' + file_name + cds[i] + '.npy'
    
    file_name = 'indp_' + cds[i]
    # array_id_IAV_reassortant_indp_pb2
    freq_path = '../Res/model_evaluate/lucaOne/new_AIV_all_8_ha_' + file_name + '_protein_emb_lucaone_array_0503_ordered.npy'
    label_path = '../Res/npy/array_label_IAV_reassortant_indp_pb2.npy'
    id_path = '../Res/npy/array_id_IAV_reassortant_indp_pb2.npy'

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
        torch.load('../Res/pt/resnet_classification_ha_sampled5000_' + cds[i] + '_lucaOne_1150+0.03_resnet34.pt'))
    # '../Res/pt/resnet_classification_' + file_name + '_lucaOne_' + str(mdl_batch_size) + '+' + str(lr) + '+' + str(N_EPOCHS * n_splits) + '_resnet34.pt')
    model.to(device)
    
    file_name += '_30'
    id, pred, true, probs = test(model, loader)
    fp = 0
    fn = 0
    num = 0
    label_lst = []
    pred_lst = [pred_.item() for pred_ in pred]
    probs_lst = [prob.item() for prob in probs]
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
    df['prob'] = probs_lst
    df['test_label'] = label_lst
    df.to_csv('../Res/prediction_result_csv/test_' + file_name + '_lucaOne_1150+0.03_resnet34.csv',
              index=False)
