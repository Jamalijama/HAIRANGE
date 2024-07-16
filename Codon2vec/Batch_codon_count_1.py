import os

import numpy as np
import pandas as pd

from Counting_codons import CodonCounting

path = '../DataCleaning/Res/res1/'
# file_name = 'AIV_all_8_before2020_29634_pb2'
# res_name = 'before2020_29634_pb2'
# file_name = 'IAV_reassortant_1450to2500_new_4_indp_pb2'
# res_name = 'IAV_reassortant_indp_pb2'
# file_name = 'AIV_all_8_after2020_27016_pb2'
# res_name = 'after2020_27016_pb2'
# file_name = 'AIV_23366_all_AvianSwine_pb2'
# res_name = '23366_all_AvianSwine_pb2'
# file_name = 'AIV_reassort_84_pb2'
# res_name = 'AIV_reassort_84_pb2'
# file_name = 'test_pb2'
# res_name = 'test_pb2'
#file_name = 'AIV_all_8_avian_humanH3N2_pb2'
#res_name = 'AIV_all_8_avian_humanH3N2_pb2'
#file_name = 'H7N9_H5N1_all_4_pb2'
#res_name = 'H7N9_H5N1_all_4_pb2'
file_name = 'AIV_all_8_avian_humanH1N1_pb2'
res_name = 'AIV_all_8_avian_humanH1N1_pb2'


file_list = os.listdir(path)

target_cols = ['seq']

seqlen_max = [760, 758, 717, 499]
seqlen_opt = [768, 768, 768, 512]

for file_obj in file_list:
    if file_obj.startswith(file_name) & file_obj.endswith('.csv'):
        print(file_obj)
        csv_data = pd.read_csv(path + file_obj)
        print(csv_data.shape)

        seq_num = csv_data.shape[0]
        id_lst = csv_data.loc[:, 'id'].tolist()
        label_lst = csv_data.loc[:, 'label'].tolist()
        i = 0

        for target_col in target_cols:
            dnt_count_list = []
            df_NCR = pd.DataFrame()
            array_count = np.zeros(shape=(seq_num, 1))
            orf_seq_list = csv_data.loc[:, target_col].tolist()

            num = 192

            count_all = []
            ids = []
            labels = []
            for seq_i in range(seq_num):
                my_seq = orf_seq_list[seq_i].lower()
                seq_len = (len(my_seq))
                test = []
                count_seq = []

                for l in range(0, seq_len, 3):

                    if l >= num:
                        seqcut0 = my_seq[l - num: l]
                    else:
                        seqcut0 = my_seq[-(num - l):] + my_seq[: l]

                    if l <= seq_len - num:  # l + num <= seqlen
                        seqcut1 = my_seq[l: l + num]
                    else:
                        seqcut1 = my_seq[l:seq_len] + my_seq[: (num - seq_len + l)]

                    seqcut = seqcut0 + seqcut1
                    counting = CodonCounting(seqcut)
                    count_seq.append(counting)
                if len(count_seq) == seqlen_max[0]:
                    for _ in range(seqlen_opt[0] - seqlen_max[0]):
                        count_seq.append(np.zeros(64))
                    count_all.append(count_seq)
                    ids.append(id_lst[seq_i])
                    labels.append(label_lst[seq_i])

            array_count_all = np.array(count_all)
            print(array_count_all.shape)
            array_labels = np.array(labels)
            array_id_all = np.array(ids)

            np.save('../Res/npy/array_codon_freq_' + res_name + '.npy', array_count_all, allow_pickle=True)
            np.save('../Res/npy/array_label_' + res_name + '.npy', array_labels, allow_pickle=True)
            np.save('../Res/npy/array_id_' + res_name + '.npy', array_id_all, allow_pickle=True)
            i += 1
