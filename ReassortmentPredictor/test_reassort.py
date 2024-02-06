from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
# sequences after 2020 for testing
# file_name = 'after2020_27016'

# given reassort sequences
# file_name = 'AIV_reassort_84'

# sequences for simulated sequence reassortment
file_name = '23366_all_AvianSwine'

freq_path = '../Res/npy_reassort/array_codon_freq_' + file_name + '.npy'
label_path = '../Res/npy/array_label_' + file_name + '_pb2.npy'
id_path = '../Res/npy/array_id_' + file_name + '_pb2.npy'

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
loader = Data.DataLoader(MyDataSet_id(inputs, labels, ids), mdl_batch_size, False, num_workers=0)

model = ResNet34(2, 6)
model.load_state_dict(torch.load('../Res/pt/resnet_classification_500+0.03_resnet34_reassort.pt'))
model.to(device)
id, pred, true, probs = test(model, loader)
fp = 0
fn = 0
num = 0
label_lst = []
pred_lst = [pred_.item() for pred_ in pred]
probs_lst = [prob.item() for prob in probs]
for i in range(len(true)):
    if true[i] == 1 and pred[i].item() == 0:
        fp += 1
    elif true[i] == 0 and pred[i].item() == 1:
        fn += 1
    if true[i] != pred[i].item():
        num += 1
        print('id:', id[i])
        print('pred:', pred[i].item())
        print('true:', true[i])
        print('prob:', probs[i].item())
        label_lst.append(1)
    else:
        label_lst.append(0)
print('fp:', fp / seq_num, ' fn:', fn / seq_num)
print(1 - num / seq_num)

# print(len(true))
# print(len(probs))

df = pd.DataFrame()
df['id'] = id
df['true'] = true
df['pred'] = pred_lst
df['prob'] = probs_lst
df['test_label'] = label_lst

df.to_csv('../Res/prediction_result_csv/test_500+0.03_resnet34_reassort_' + file_name + '.csv', index=False)
