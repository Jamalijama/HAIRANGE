import pandas as pd
from sklearn.utils import shuffle

dir1 = './Res/res1/'


def JudgeSeq(seq):
    num = 0
    flag = 0
    for i in range(len(seq)):
        if seq[i] == '-':
            num += 1
        if seq[i] == 'n':
            flag = 1
            break
    return num / len(seq), flag


def SplitYear(df, file):
    df = df[(df['Host_IRD'] == 0) | (df['Host_IRD'] == 1)]
    df_1 = df[df['Year'] < 2020]
    df_2 = df[df['Year'] >= 2020]
    df_1.to_excel(file + str(df_1.shape[0]) + '_before2020.xlsx', index=False)
    df_2.to_excel(file + str(df_2.shape[0]) + '_after2020.xlsx', index=False)


def Split4(df, file):
    cols = ['pb2', 'pb1', 'pa', 'np']
    strain_names = df['Strain_name'].tolist()
    serotypes = df['Serotype'].tolist()
    Host0s = df['Host0'].tolist()
    Hosts = df['Host'].tolist()
    years = df['Year'].tolist()
    countries = df['Country'].tolist()
    Host_irds = df['Host_IRD'].tolist()
    continents = df['Continent'].tolist()
    pb2 = df['CDS_1'].tolist()
    pb1 = df['CDS_2'].tolist()
    pa = df['CDS_3'].tolist()
    np = df['CDS_5'].tolist()

    seq_lst = []
    seq_lst.append(pb2)
    seq_lst.append(pb1)
    seq_lst.append(pa)
    seq_lst.append(np)

    seqID_lst = []
    seqLabel_lst = []

    for i in range(df.shape[0]):
        seqID = strain_names[i] + '$' + str(serotypes[i]) + '$' + str(Host0s[i]) + '$' + str(years[i]) + '$' + str(
            countries[i]) + '$' + str(Hosts[i]) + '$' + str(continents[i])
        seqID_lst.append(seqID)
        seqLabel_lst.append(Host_irds[i])

    for i in range(4):
        df_1 = pd.DataFrame()
        df_1['id'] = seqID_lst
        df_1['seq'] = seq_lst[i]
        df_1['label'] = seqLabel_lst
        df_1.to_csv(file + cols[i] + '.csv', index=False)


def SampleHuman(df, file):
    df_avian = df[df['Host_IRD'] == 0]
    df_human = df[df['Host_IRD'] == 1]
    print(df_avian.shape[0])
    df_human_sample = df_human.sample(n=df_avian.shape[0])
    df_sampled = pd.concat([df_human_sample, df_avian], random_state=27)
    data = df_sampled.values
    data = shuffle(data, random_state=27)
    df_sampled = pd.DataFrame(data, columns=df_sampled.columns)
    # print(df_sampled.shape)
    df_sampled.to_excel(file + str(df_sampled.shape[0]) + '.xlsx', index=False)


if __name__ == '__main__':
    # split data by year
    file = dir1 + 'new_AIV_all_8_73805.xlsx'
    df = pd.read_excel(file, sheet_name='AIV_all_8_73805')
    SplitYear(df, dir1 + 'AIV_all_8_')

    # sample human data before 2020
    file = dir1 + 'AIV_all_8_39955_before2020.xlsx'
    df = pd.read_excel(file)
    SampleHuman(df, dir1 + 'AIV_all_8_before2020_')

    # split 4 segments
    file = dir1 + 'AIV_all_8_before2020_29634.xlsx'
    df = pd.read_excel(file)
    Split4(df, dir1 + 'AIV_all_8_before2020_29634_')
    #
    file = dir1 + 'AIV_all_8_27016_after2020.xlsx'
    df = pd.read_excel(file)
    Split4(df, dir1 + 'AIV_all_8_after2020_27016_')

    file = dir1 + 'AIV_reassort_84.xlsx'
    df = pd.read_excel(file)
    Split4(df, dir1 + 'AIV_reassort_84_')
