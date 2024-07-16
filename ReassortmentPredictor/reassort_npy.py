from module import *
from resnet_18_34 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cds = ['pb2', 'pb1', 'pa', 'np']
dir = ['', '_trm', '', '_trm']
seqlen_max = [760, 758, 717, 499]
#threshold = [0.5, 0.3, 0.35, 0.55]
#Maxlen = 1536
#threshold = [0.55, 0.05, 0.05, 0.6]
#Maxlen = 1280
threshold = [0, 0, 0, 0]
Maxlen = 2816


def StackReassort(dir1, file_lst, file_name):
    inputs1 = []
    for i in range(len(file_lst)):
        coden_loc_lst = []
        df = pd.read_excel(dir1 + file_lst[i])
        avg_importance_lst = df['avg_norm'].tolist()
        for j in range(len(avg_importance_lst)):
            if avg_importance_lst[j] >= threshold[i]:
                coden_loc_lst.append(j)

        inputs = np.load('../Res/npy' + dir[i] + '/array_codon_freq_' + file_name + cds[i] + dir[i] + '.npy',
                         allow_pickle=True)
        tmp_inputs = []
        for j in range(inputs.shape[0]):
            seq_npy = []
            for k in range(inputs.shape[1]):
                if k in coden_loc_lst:
                    seq_npy.append(inputs[j, k, :])
            tmp_inputs.append(seq_npy)
        print(np.array(tmp_inputs).shape)
        inputs1.append(np.array(tmp_inputs))

    inputs = np.hstack((inputs1[0], inputs1[1], inputs1[2], inputs1[3]))
    print(inputs.shape)

    inputs2 = []
    for i in range(inputs.shape[0]):
        #    print(inputs[i].shape)
        length = len(inputs[i])
        tmp_inputs = np.vstack((inputs[i], np.zeros((Maxlen - length, 64))))
        inputs2.append(tmp_inputs)
    print(len(inputs2[0]))

    np.save('../Res/npy_reassort/array_codon_freq_' + file_name[:-1] + '_0701_2816.npy', inputs2, allow_pickle=True)


dir1 = '../Res/codon_importance/'
file_lst = ['pb2_80_1150+0.03_0.2_resnet34_new.xlsx', 'pb1_80_1150+0.03_0.2_resnet34_trm_new.xlsx',
            'pa_80_1150+0.03_0.2_resnet34_new.xlsx', 'np_80_1150+0.03_0.2_resnet34_trm_new.xlsx']

# sequences before 2020 for training single model
#file_name = 'before2020_29634_'
#StackReassort(dir1, file_lst, file_name)
##
## sequences after 2020 for testing single model
#file_name = 'after2020_27016_'
#StackReassort(dir1, file_lst, file_name)
#
## given reassort sequences
#file_name = 'AIV_reassort_84_'
#StackReassort(dir1, file_lst, file_name)
#
## sequences for simulated sequence reassortment
#file_name = '23366_all_AvianSwine_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'IAV_reassortant_indp_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'test_'
#StackReassort(dir1, file_lst, file_name)
#
#file_name = 'AIV_all_8_avian_humanH3N2_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'H7N9_H5N1_all_4_'
#StackReassort(dir1, file_lst, file_name)

#file_name = 'AIV_all_8_avian_humanH1N1_'
#StackReassort(dir1, file_lst, file_name)

file_name = 'AIV_all_8_a_hH1N1_aH3N2_0.05_'
StackReassort(dir1, file_lst, file_name)