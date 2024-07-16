import numpy as np
import pandas as pd

dir1 = './Res/mafft/'
dir2 = './Res/final/'
file_lst = ['pb2', 'pb1', 'pa', 'np']
cds_lst_name = ['CDS_1', 'CDS_2', 'CDS_3', 'CDS_5']
cds_len_name = ['CDS_len_1', 'CDS_len_2', 'CDS_len_3', 'CDS_len_5']


# Fasta file after mafft
def FastaToCsv():
    for path in file_lst:
        df = pd.DataFrame()
        seq_id_lst = []
        seq_lst = []
        seq = ''
        flag = 0
        with open(dir1 + path + '.fasta', "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                if line[0] == ">":
                    if flag == 1:
                        seq_lst.append(seq)
                        seq = ''
                        flag = 0
                    seq_id_lst.append(line)
                else:
                    seq += line
                    flag = 1
            seq_lst.append(seq)

        df['seq_id'] = seq_id_lst
        df['seq'] = seq_lst
        df.to_csv(dir1 + path + '.csv', index=False)


def FindDelCol():
    thresh = 50
    for path in file_lst:
        df = pd.read_csv(dir1 + path + '.csv')
        df1 = pd.DataFrame()
        seq_lst = df['seq']
        seq_id_lst = df['seq_id']
        print('before:', len(seq_lst[0]))

        del_col_lst = []
        total_row = len(seq_lst)
        total_col = len(seq_lst[0])
        for i in range(total_col):
            count = 0
            for j in range(total_row):
                if seq_lst[j][i] == '-':
                    count += 1
            if total_row - count <= thresh:
                del_col_lst.append(i)
                print(i)
        df1['del'] = del_col_lst
        df1.to_csv(dir1 + path + '_del_col.csv', index=False)


def DelSeqAndEmptyCol():
    i_lst = []
    for path in file_lst:
        df = pd.read_csv(dir1 + path + '.csv')
        seq_lst = df['seq'].tolist()
        seq_id_lst = df['seq_id'].tolist()
        total_row = len(seq_lst)
        total_col = len(seq_lst[0])
        df1 = pd.read_csv(dir1 + path + '_del_col.csv')
        del_col_lst = df1['del'].tolist()
        i = 0
        while i < total_row:
            str = seq_lst[i]
            for j in del_col_lst:
                if str[j] != '-':
                    i_lst.append(i)
                    seq_lst.pop(i)
                    seq_id_lst.pop(i)
                    total_row = len(seq_lst)
                    i -= 1
                    break
            i += 1

        seq_char_lst = []
        for i in seq_lst:
            seq = list(i)
            seq_char_lst.append(seq)

        seq_arr = np.array(seq_char_lst)
        print('before:', seq_arr.shape)
        total_row = seq_arr.shape[0]
        total_col = seq_arr.shape[1]

        i = 0
        while i < total_col:
            count = 0
            for j in range(total_row):
                if seq_arr[j][i] == '-':
                    count += 1
            if count == total_row:
                seq_arr = np.delete(seq_arr, i, axis=1)
                total_col = seq_arr.shape[1]
            else:
                i += 1
            print(i)

        print('after:', seq_arr.shape)

        seq_lst = []
        for i in range(seq_arr.shape[0]):
            str = ''.join(seq_arr[i])
            seq_lst.append(str)

        df2 = pd.DataFrame()
        df2['seq_id'] = seq_id_lst
        df2['seq'] = seq_lst
        df2.to_csv(dir1 + path + '_del_empty.csv', index=False)
    return i_lst


def MergeSeqInf():
    start_lst = [32, 29, 27, 50]
    end_lst = [2312, 2302, 2177, 1546]
    CDS_lst = [[] for _ in range(4)]
    CDS_len_lst = [[] for _ in range(4)]

    for i in range(len(file_lst)):
        if i == 0:
            file = dir2 + 'AIV_all_8_79630.csv'
        else:
            file = dir2 + 'AIV_all_8_79630_' + str(i) + '.csv'
        df = pd.read_csv(file)
        df_ = pd.read_csv(dir1 + file_lst[i] + '_del_empty.csv')
        seq_id_lst = [str[1:] for str in df_['seq_id']]
        strain_lst = df['Strain_name']
        seq_lst = df_['seq']
        k = 0
        all = len(df['Strain_name'])
        while k < all:
            flag = 1
            j = 0
            while j < len(seq_lst):
                if strain_lst[k] == seq_id_lst[j]:
                    flag = 0
                    CDS_lst[i].append(seq_lst[j][start_lst[i] - 1:end_lst[i]])
                    CDS_len_lst[i].append(len(seq_lst[j][start_lst[i] - 1:end_lst[i]]))
                    break
                else:
                    j += 1
            if flag == 1:
                df.drop(index=[k], inplace=True)
                print(df.shape)
                print(k)
            k += 1
        df[cds_lst_name[i]] = CDS_lst[i]
        df[cds_len_name[i]] = CDS_len_lst[i]
        df.to_csv(dir2 + 'AIV_all_8_79630_' + str(i + 1) + '.csv', index=False)


def DelPB2Empty():
    file = dir2 + 'AIV_all_8_79630_4.csv'
    df = pd.read_csv(file)
    seq_lst = df['CDS_1'].tolist()
    seq_lst_new = []
    seq_len_lst = []
    for s in seq_lst:
        if s[128] == '-':
            seq_lst_new.append(s[:128] + s[129:])
            seq_len_lst.append(2280)
        elif s[129] == '-':
            seq_lst_new.append(s[:129] + s[130:])
            seq_len_lst.append(2280)
    df['CDS_1'] = seq_lst_new
    df['CDS_len_1'] = seq_len_lst
    # 79181
    df.to_csv(dir2 + 'AIV_all_8_' + str(len(seq_lst)) + '.csv', index=False)


def DelBeginAndEnd():
    begin = ['atg']
    end = ['taa', 'tag', 'tga']
    df = pd.read_csv(dir2 + 'AIV_all_8_79181.csv')
    del_lst = []
    for i in range(df.shape[0]):
        if df['CDS_1'][i][:3] in begin and df['CDS_1'][i][-3:] in end \
                and df['CDS_2'][i][:3] in begin and df['CDS_2'][i][-3:] in end \
                and df['CDS_3'][i][:3] in begin and df['CDS_3'][i][-3:] in end \
                and df['CDS_5'][i][:3] in begin and df['CDS_5'][i][-3:] in end:
            continue
        else:
            print(i)
            del_lst.append(i)
    df.drop(df.index[del_lst], inplace=True)
    print(df.shape)
    # 73805
    df.to_csv(dir2 + 'AIV_all_8_' + str(df.shape[0]) + '.csv', index=False)


if __name__ == '__main__':
    # Cleaning results after mafft alignment
    #    FastaToCsv()
    #    FindDelCol()
    #    DelSeqAndEmptyCol()
    #    MergeSeqInf()
    #    DelPB2Empty()
    DelBeginAndEnd()
