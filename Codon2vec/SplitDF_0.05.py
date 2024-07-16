import numpy as np
import pandas as pd


file = '../DataCleaning/Res/res1/AIV_all_8_avian_humanH3N2.xlsx'
df = pd.read_excel(file)
df1 = df[(df['Host'] == 'Human')].sample(n=560, random_state=7)
#print(df1.index.tolist())

cds = ['pb2', 'pb1', 'pa', 'np']
file_name1 = 'avian_humanH1N1_'
file_name2 = 'avian_humanH3N2_'
res_name = 'a_hH1N1_aH3N2_0.05_'

for i in range(len(cds)):
    freq_path1 = '../Res/npy/array_codon_freq_AIV_all_8_' + file_name1 + cds[i] + '.npy'
    label_path1 = '../Res/npy/array_label_AIV_all_8_' + file_name1 + cds[i] + '.npy'

    inputs1 = np.load(freq_path1, allow_pickle=True)
    labels1 = np.load(label_path1, allow_pickle=True)

    freq_path2 = '../Res/npy/array_codon_freq_AIV_all_8_' + file_name2 + cds[i] + '.npy'
    label_path2 = '../Res/npy/array_label_AIV_all_8_' + file_name2 + cds[i] + '.npy'

    inputs2 = np.load(freq_path2, allow_pickle=True)
    labels2 = np.load(label_path2, allow_pickle=True)

    list_inputs1 = list(inputs1)
    list_labels1 = list(labels1)

    for j in df1.index.tolist():
#        print(j)
        list_inputs1.append(inputs2[j, :, :])
        list_labels1.append(labels2[j])

    array_inputs = np.array(list_inputs1)
    array_labels = np.array(list_labels1)
    
    print(array_inputs.shape)
    print(array_labels.shape)

    np.random.seed(7)
    np.random.shuffle(array_inputs)
    np.random.seed(7)
    np.random.shuffle(array_labels)
    
    print(array_labels)

    np.save('../Res/npy/array_codon_freq_AIV_all_8_' + res_name + cds[i] +'.npy', array_inputs, allow_pickle=True)
    np.save('../Res/npy/array_label_AIV_all_8_' + res_name + cds[i] + '.npy', array_labels, allow_pickle=True)

