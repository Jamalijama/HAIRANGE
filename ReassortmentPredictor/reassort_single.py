from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cds = ['pb2', 'pb1', 'pa', 'np']
dir = ['', '_trm', '', '_trm']
seqlen_max = [760, 758, 717, 499]
threshold = [0.45, 0.45, 0.45, 0.45]
dir1 = '../Res/codon_importance/'
file_lst = ['pb2_80_1150+0.03_0.2_resnet34_new.xlsx', 'pb1_80_1150+0.03_0.2_resnet34_trm_new.xlsx',
            'pa_80_1150+0.03_0.2_resnet34_new.xlsx', 'np_80_1150+0.03_0.2_resnet34_trm_new.xlsx']

# sequences before 2020 for training single model
#file_name = 'before2020_29634_'

# sequences for simulated sequence reassortment
#file_name = '23366_all_AvianSwine_'

file_name = 'H7N9_H5N1_all_4_'

for i in range(len(file_lst)):
    coden_loc_lst = []
    df = pd.read_excel(dir1 + file_lst[i])
    loc_lst = df['loc'].tolist()
    avg_importance_lst = df['avg_norm'].tolist()
    for j in range(len(avg_importance_lst)):
        if avg_importance_lst[j] >= threshold[i]:
            coden_loc_lst.append(j)
    print(len(coden_loc_lst))

    inputs = np.load('../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy',
                     allow_pickle=True)
    inputs1 = []
    for j in range(inputs.shape[0]):
        seq_npy = []
        for k in range(inputs.shape[1]):
            if k in coden_loc_lst:
                seq_npy.append(inputs[j, k, :])
        inputs1.append(seq_npy)
    print(len(inputs1[0]))
    inputs1 = np.array(inputs1)
    np.save('../Res/npy_reassort/array_codon_freq_' + file_name + cds[i] + dir[i] + '_nofilled.npy', inputs1,
            allow_pickle=True)
