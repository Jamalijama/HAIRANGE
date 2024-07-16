import numpy as np
import pandas as pd

cds = ['pb2', 'pb1', 'pa', 'np']
dir = ['', '_trm', '', '_trm']


file_name = 'H7N9_H5N1_all_4_'
file1 = '../DataCleaning/Res/res1/H7N9_H5N1_all_4.xlsx'

lst_num = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
           [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
           [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]

df = pd.read_excel(file1)
lst_name = df['isolate_name'].tolist()
lst_host0 = df['host0'].tolist()
lst_host = df['host'].tolist()
lst_subtype = df['subtype'].tolist()
lst_year = df['year'].tolist()
lst_country = df['country'].tolist()
lst_continent = df['continent'].tolist()

line_human = -1
# H3N2
if 'A/Kansas/14/2017' in lst_name:
    line_human = lst_name.index('A/Kansas/14/2017')


line_avian = []
for i in range(len(lst_host)):
    if lst_host[i] == 'Avian' or lst_host[i] == 'Swine' or lst_host[i] == 'Human' and i != line_human:
        line_avian.append(i)

lst4 = []
for i in range(len(cds)):
    matrix = np.load('../Res/npy_reassort/array_codon_freq_' + file_name + cds[i] + dir[i] + '_nofilled_0603_1536.npy',
                     allow_pickle=True)
    lst4.append(matrix)

lst_sname = []
lst_matrix = []
for i in line_avian:
    for lst_j in lst_num:
        sname = ''
        lst_tmp = []
        for j in range(len(lst_j)):
            if lst_j[j] == 1:
                sname += cds[j] + '#' + lst_name[line_human] + '#' + lst_host0[line_human] + '#' + lst_host[
                    line_human] + '#' + lst_subtype[line_human] + '#' + str(lst_year[line_human]) + '#' + str(
                    lst_country[line_human]) + '#' + str(lst_continent[line_human]) + '$$$'
                lst_tmp.append(lst4[j][line_human])
            else:
                sname += cds[j] + '#' + lst_name[i] + '#' + str(lst_host0[i]) + '#' + lst_host[i] + '#' + str(
                    lst_subtype[
                        i]) + '#' + str(lst_year[i]) + '#' + str(lst_country[i]) + '#' + str(lst_continent[i]) + '$$$'
                lst_tmp.append(lst4[j][i])
        lst_sname.append(sname)
        length = lst_tmp[0].shape[0] + lst_tmp[1].shape[0] + lst_tmp[2].shape[0] + lst_tmp[3].shape[0]
        matrix_tmp = np.vstack((lst_tmp[0], lst_tmp[1], lst_tmp[2], lst_tmp[3], np.zeros((1536 - length, 64))))
        lst_matrix.append(matrix_tmp)

df1 = pd.DataFrame()
df1['name'] = lst_sname
df1.to_csv('../Res/res1/humanH3N2_AvianSwine_simulated_name_' + file_name[:-1] + '_0603_1536.csv', index=False)

array_matrix = np.array(lst_matrix)
print(array_matrix.shape)
np.save('../Res/npy_reassort/humanH3N2_AvianSwine_simulated_' + file_name[:-1] + '_0603_1536.npy', array_matrix,
        allow_pickle=True)
