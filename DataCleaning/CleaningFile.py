import numpy as np
import pandas as pd
from Bio import SeqIO

dir1 = './Res/BVBRC/'
dir2 = './Res/GISAID/'
dir3 = './Res/NCBI/'
dir4 = './Res/final/'
dir5 = './Res/fasta/'


def CutFasta():
    path_lst = ['./AIV_2021_10_22to2023/BVBRC/BVBRC_genome_sequence.fasta',
                './AIV_2021_10_22to2023/GISAID/2021_10_22 gisaid_epiflu_sequence.fasta',
                './AIV_2021_10_22to2023/GISAID/2022_03_22 gisaid_epiflu_sequence.fasta',
                './AIV_2021_10_22to2023/GISAID/2022_10_22 gisaid_epiflu_sequence.fasta']

    for path in path_lst:
        df = pd.DataFrame()
        seq_id_lst = []
        seq_lst = []
        seq = ''
        flag = 0
        with open(path, "r") as f:
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
            seq = ''

        df['seq_id'] = seq_id_lst
        df['seq'] = seq_lst
        df.to_csv(dir1 + path.split('/', 3)[3].split('.')[0] + '.csv', index=False)


def CutBVBRCSeqId():
    file = 'BVBRC_genome_sequence.csv'
    csv_file = pd.read_csv(dir1 + file)

    seq_id_lst = csv_file.seq_id.tolist()
    seq_lst = csv_file.seq.tolist()

    df = pd.DataFrame()
    genbank_id = []
    segment_id = []
    genome_name = []
    seq = []

    for i in range(len(seq_id_lst)):
        id_lst = seq_id_lst[i].split('  ')
        temp = id_lst[1].split(' ')
        # print(i, id_lst[1])
        flag = 0
        temp_len = len(temp)
        for j in range(temp_len):
            if temp[j] == 'segment':
                flag = 1
                segment_id.append(int(temp[j + 1].strip(',')))
        if flag == 1:
            genome_name.append(id_lst[2].split(' ', 4)[-1].split(' | ')[0])
            genbank_id.append(id_lst[0].split('|')[-1])
            seq.append(seq_lst[i])

    df['genome_name'] = genome_name
    df['segment_id'] = segment_id
    df['genbank_id'] = genbank_id
    df['seq'] = seq
    data = df.sort_values(by=['genome_name', 'segment_id', 'genbank_id'], ascending=True)
    data.to_csv(dir1 + "AIV_2021_10_22to2023_BVBRC_fasta_CutSeqId.csv", index=False)


def ExtractBVBRCXlsx():
    excel_file = './AIV_2021_10_22to2023/BVBRC/BVBRC_genome.xlsx'

    data = pd.read_excel(excel_file, usecols=['Strain', 'Segment', 'GenBank Accessions',
                                              'Subtype', 'Host Name', 'Host Common Name',
                                              'Collection Date', 'Collection Year',
                                              'Isolation Country', 'Geographic Group'])

    df = data.sort_values(by=['Strain', 'Segment', 'GenBank Accessions'], ascending=True)
    df.to_csv(dir1 + "AIV_2021_10_22to2023_BVBRC_xlsx.csv", index=False)


def CombineBVBRCFastaAndXlsx():
    file_name1 = 'AIV_2021_10_22to2023_BVBRC_fasta_CutSeqId.csv'
    df1 = pd.read_csv(dir1 + file_name1)
    file_name2 = 'AIV_2021_10_22to2023_BVBRC_xlsx.csv'
    df2 = pd.read_csv(dir1 + file_name2)
    mammal_lst = ['Bottlenose Dolphin', 'Dog', 'Cat', 'Horse']

    strain_lst = []
    subtype_lst = []
    host_lst = []
    host0_lst = []
    seq_len_lst = []
    date_lst = []
    year_lst = []
    country_lst = []
    continent_lst = []

    i = 0
    j = 0
    df2['Collection Date'] = pd.to_datetime(df2['Collection Date'])
    while i < len(df1['genome_name']) and j < len(df2['Strain']):
        if df1['genome_name'][i] == df2['Strain'][j] \
                and df1['genbank_id'][i] == df2['GenBank Accessions'][j] \
                and df1['segment_id'][i] == df2['Segment'][j]:
            strain_lst.append(df2['Strain'][j])
            subtype_lst.append(df2['Subtype'][j])
            host0_lst.append(df2['Host Common Name'][j])
            if 'Human' in df2['Host Common Name'][j]:
                host_lst.append('Human')
            elif 'Pig' in df2['Host Common Name'][j]:
                host_lst.append('Swine')
            elif df2['Host Common Name'][j] in mammal_lst:
                host_lst.append('Mammal')
            else:
                host_lst.append('Avian')
            date_lst.append(df2['Collection Date'][j])
            year_lst.append(df2['Collection Year'][j])
            country_lst.append(df2['Isolation Country'][j])
            continent_lst.append(df2['Geographic Group'][j])
            seq_len_lst.append(len(df1['seq'][i]))
            i += 1
            j += 1
        else:
            j += 1
    df1['seq_len'] = seq_len_lst
    df1['subtype'] = subtype_lst
    df1['host'] = host_lst
    df1['host0'] = host0_lst
    df1['date'] = date_lst
    df1['year'] = year_lst
    df1['country'] = country_lst
    df1['continent'] = continent_lst
    df1.to_csv(dir1 + 'AIV_2021_10_22to2023_BVBRC_fasta_xlsx.csv', index=False)


def MergeBVBRC8Seg():
    path = dir1 + 'AIV_2021_10_22to2023_BVBRC_fasta_xlsx.csv'
    df = pd.read_csv(path)

    df1 = pd.DataFrame()
    strain_lst = []
    subtype_lst = []
    host_lst = []
    host0_lst = []
    date_lst = []
    year_lst = []
    country_lst = []
    continent_lst = []
    source_lst = []

    CDS1_lst = []
    CDS1_len_lst = []
    CDS2_lst = []
    CDS2_len_lst = []
    CDS3_lst = []
    CDS3_len_lst = []
    CDS4_lst = []
    CDS4_len_lst = []
    CDS5_lst = []
    CDS5_len_lst = []
    CDS6_lst = []
    CDS6_len_lst = []
    CDS7_lst = []
    CDS7_len_lst = []
    CDS8_lst = []
    CDS8_len_lst = []

    flag = np.zeros(8)
    i = 0
    strain = ''
    while i < df.shape[0]:
        if 0 not in flag:
            if df['segment_id'][i] == 1:
                flag = np.zeros(8)
                i -= 1
        elif flag[0] == 0 and df['segment_id'][i] == 1:
            strain = df['genome_name'][i]
            strain_lst.append(df['genome_name'][i])
            subtype_lst.append(df['subtype'][i])
            host_lst.append(df['host'][i])
            host0_lst.append(df['host0'][i])
            date_lst.append(df['date'][i])
            year_lst.append(df['year'][i])
            country_lst.append(df['country'][i])
            continent_lst.append(df['continent'][i])
            source_lst.append('BVBRC')

            flag[0] = 1
            CDS1_lst.append(df['seq'][i].upper())
            CDS1_len_lst.append(df['seq_len'][i])
        elif flag[1] == 0 and df['segment_id'][i] == 2:
            flag[1] = 1
            CDS2_lst.append(df['seq'][i].upper())
            CDS2_len_lst.append(df['seq_len'][i])
        elif flag[2] == 0 and df['segment_id'][i] == 3:
            flag[2] = 1
            CDS3_lst.append(df['seq'][i].upper())
            CDS3_len_lst.append(df['seq_len'][i])
        elif flag[3] == 0 and df['segment_id'][i] == 4:
            flag[3] = 1
            CDS4_lst.append(df['seq'][i].upper())
            CDS4_len_lst.append(df['seq_len'][i])
        elif flag[4] == 0 and df['segment_id'][i] == 5:
            flag[4] = 1
            CDS5_lst.append(df['seq'][i].upper())
            CDS5_len_lst.append(df['seq_len'][i])
        elif flag[5] == 0 and df['segment_id'][i] == 6:
            flag[5] = 1
            CDS6_lst.append(df['seq'][i].upper())
            CDS6_len_lst.append(df['seq_len'][i])
        elif flag[6] == 0 and df['segment_id'][i] == 7:
            flag[6] = 1
            CDS7_lst.append(df['seq'][i].upper())
            CDS7_len_lst.append(df['seq_len'][i])
        elif flag[7] == 0 and df['segment_id'][i] == 8:
            flag[7] = 1
            CDS8_lst.append(df['seq'][i].upper())
            CDS8_len_lst.append(df['seq_len'][i])
        i += 1

    df1['Strain_name'] = strain_lst
    df1['CDS_1'] = CDS1_lst
    df1['CDS_2'] = CDS2_lst
    df1['CDS_3'] = CDS3_lst
    df1['CDS_4'] = CDS4_lst
    df1['CDS_5'] = CDS5_lst
    df1['CDS_6'] = CDS6_lst
    df1['CDS_7'] = CDS7_lst
    df1['CDS_8'] = CDS8_lst
    df1['CDS_len_1'] = CDS1_len_lst
    df1['CDS_len_2'] = CDS2_len_lst
    df1['CDS_len_3'] = CDS3_len_lst
    df1['CDS_len_4'] = CDS4_len_lst
    df1['CDS_len_5'] = CDS5_len_lst
    df1['CDS_len_6'] = CDS6_len_lst
    df1['CDS_len_7'] = CDS7_len_lst
    df1['CDS_len_8'] = CDS8_len_lst
    df1['Host'] = host_lst
    df1['Host0'] = host0_lst
    df1['Serotype'] = subtype_lst
    df1['Date'] = date_lst
    df1['Year'] = year_lst
    df1['Country'] = country_lst
    df1['Continent'] = continent_lst
    df1['data_label'] = source_lst
    df1.to_csv(dir1 + 'AIV_2021_10_22to2023_BVBRC_fasta_xlsx_8.csv', index=False)


def CutGISAIDId():
    file_lst = [dir1 + '2021_10_22 gisaid_epiflu_sequence.csv',
                dir1 + '2022_03_22 gisaid_epiflu_sequence.csv',
                dir1 + '2022_10_22 gisaid_epiflu_sequence.csv']
    for file in file_lst:
        csv_file = pd.read_csv(file)
        seq_id_lst = csv_file.seq_id.tolist()
        seq_lst = csv_file.seq.tolist()

        df = pd.DataFrame()
        isolate_id = []
        isolate_name = []
        subtype = []
        segment = []
        unique_id = []
        seq_len = []
        for i in range(len(seq_id_lst)):
            id_lst = seq_id_lst[i].split('|')
            isolate_id.append(id_lst[1])
            isolate_name.append(id_lst[0].strip('>'))
            subtype.append(id_lst[2].split('_')[-1])
            segment.append(id_lst[-2])
            unique_id.append(id_lst[-1])
            seq_len.append(len(seq_lst[i]))

        df['isolate_name'] = isolate_name
        df['unique_id'] = unique_id
        df['isolate_id'] = isolate_id
        df['segment'] = segment
        df['subtype'] = subtype
        df['seq'] = seq_lst
        df['seq_len'] = seq_len
        data = df.sort_values(by=['isolate_id', 'segment'], ascending=True)
        data.to_csv(dir2 + file.split('/')[3].split('.')[0] + "_fasta_CutSeqId.csv", index=False)


def ExtractGISAIDXlsx():
    file_lst = ['./AIV_2021_10_22to2023/GISAID/2021_10_22 gisaid_epiflu_isolates.xls',
                './AIV_2021_10_22to2023/GISAID/2022_03_22 gisaid_epiflu_isolates.xls',
                './AIV_2021_10_22to2023/GISAID/2022_10_22 gisaid_epiflu_isolates.xls']

    for file in file_lst:
        data = pd.read_excel(file, usecols=['Isolate_Id', 'PB2 Segment_Id', 'PB1 Segment_Id',
                                            'PA Segment_Id', 'HA Segment_Id', 'NP Segment_Id',
                                            'NA Segment_Id', 'MP Segment_Id', 'NS Segment_Id',
                                            'Isolate_Name', 'Subtype', 'Host', 'Collection_Date',
                                            'Location'], )
        data.dropna(axis=0, how='any', subset=None, inplace=True)
        df = data.sort_values(by=['Isolate_Id'], ascending=True)
        df.to_csv(dir2 + file.split('/')[3].split('.')[0] + "_xls.csv", index=False)


def MergeGISAID8Seg():
    path_lst = [dir2 + '2021_10_22 gisaid_epiflu_sequence_fasta_CutSeqId.csv',
                dir2 + '2022_03_22 gisaid_epiflu_sequence_fasta_CutSeqId.csv',
                dir2 + '2022_10_22 gisaid_epiflu_sequence_fasta_CutSeqId.csv']
    for path in path_lst:
        df = pd.read_csv(path)

        df1 = pd.DataFrame()
        strain_lst = []
        strain_id = []
        subtype_lst = []

        CDS1_lst = []
        CDS1_len_lst = []
        unq1_lst = []
        CDS2_lst = []
        CDS2_len_lst = []
        unq2_lst = []
        CDS3_lst = []
        CDS3_len_lst = []
        unq3_lst = []
        CDS4_lst = []
        CDS4_len_lst = []
        unq4_lst = []
        CDS5_lst = []
        CDS5_len_lst = []
        unq5_lst = []
        CDS6_lst = []
        CDS6_len_lst = []
        unq6_lst = []
        CDS7_lst = []
        CDS7_len_lst = []
        unq7_lst = []
        CDS8_lst = []
        CDS8_len_lst = []
        unq8_lst = []

        flag = np.zeros(8)
        i = 0
        strain = ''
        while i < df.shape[0]:
            if df['isolate_name'][i] != strain:
                strain = df['isolate_name'][i]
                count = 1
                j = i + 1
                while j < df.shape[0] and df['isolate_name'][j] == df['isolate_name'][i]:
                    count += 1
                    j += 1
                if count < 8:
                    i = j
                    i -= 1
                else:
                    i -= 1
            else:
                if 0 not in flag:
                    if df['segment'][i] == 1:
                        flag = np.zeros(8)
                        i -= 1
                elif flag[0] == 0 and df['segment'][i] == 1:
                    strain = df['isolate_name'][i]
                    strain_lst.append(df['isolate_name'][i])
                    strain_id.append(df['isolate_id'][i])
                    subtype_lst.append(df['subtype'][i])
                    flag[0] = 1
                    unq1_lst.append(df['unique_id'][i])
                    CDS1_lst.append(df['seq'][i].upper())
                    CDS1_len_lst.append(df['seq_len'][i])
                elif flag[1] == 0 and df['segment'][i] == 2:
                    flag[1] = 1
                    unq2_lst.append(df['unique_id'][i])
                    CDS2_lst.append(df['seq'][i].upper())
                    CDS2_len_lst.append(df['seq_len'][i])
                elif flag[2] == 0 and df['segment'][i] == 3:
                    flag[2] = 1
                    unq3_lst.append(df['unique_id'][i])
                    CDS3_lst.append(df['seq'][i].upper())
                    CDS3_len_lst.append(df['seq_len'][i])
                elif flag[3] == 0 and df['segment'][i] == 4:
                    flag[3] = 1
                    unq4_lst.append(df['unique_id'][i])
                    CDS4_lst.append(df['seq'][i].upper())
                    CDS4_len_lst.append(df['seq_len'][i])
                elif flag[4] == 0 and df['segment'][i] == 5:
                    flag[4] = 1
                    unq5_lst.append(df['unique_id'][i])
                    CDS5_lst.append(df['seq'][i].upper())
                    CDS5_len_lst.append(df['seq_len'][i])
                elif flag[5] == 0 and df['segment'][i] == 6:
                    flag[5] = 1
                    unq6_lst.append(df['unique_id'][i])
                    CDS6_lst.append(df['seq'][i].upper())
                    CDS6_len_lst.append(df['seq_len'][i])
                elif flag[6] == 0 and df['segment'][i] == 7:
                    flag[6] = 1
                    unq7_lst.append(df['unique_id'][i])
                    CDS7_lst.append(df['seq'][i].upper())
                    CDS7_len_lst.append(df['seq_len'][i])
                elif flag[7] == 0 and df['segment'][i] == 8:
                    flag[7] = 1
                    unq8_lst.append(df['unique_id'][i])
                    CDS8_lst.append(df['seq'][i].upper())
                    CDS8_len_lst.append(df['seq_len'][i])
            i += 1

        df1['strain'] = strain_lst
        df1['strain_id'] = strain_id
        df1['unique_1'] = unq1_lst
        df1['unique_2'] = unq2_lst
        df1['unique_3'] = unq3_lst
        df1['unique_4'] = unq4_lst
        df1['unique_5'] = unq5_lst
        df1['unique_6'] = unq6_lst
        df1['unique_7'] = unq7_lst
        df1['unique_8'] = unq8_lst
        df1['CDS_1'] = CDS1_lst
        df1['CDS_2'] = CDS2_lst
        df1['CDS_3'] = CDS3_lst
        df1['CDS_4'] = CDS4_lst
        df1['CDS_5'] = CDS5_lst
        df1['CDS_6'] = CDS6_lst
        df1['CDS_7'] = CDS7_lst
        df1['CDS_8'] = CDS8_lst
        df1['len_CDS_1'] = CDS1_len_lst
        df1['len_CDS_2'] = CDS2_len_lst
        df1['len_CDS_3'] = CDS3_len_lst
        df1['len_CDS_4'] = CDS4_len_lst
        df1['len_CDS_5'] = CDS5_len_lst
        df1['len_CDS_6'] = CDS6_len_lst
        df1['len_CDS_7'] = CDS7_len_lst
        df1['len_CDS_8'] = CDS8_len_lst
        df1['subtype'] = subtype_lst
        df1.to_csv(dir2 + path.split('/')[3].split('.')[0] + '_8.csv', index=False)


def MergeGISAIDFastaAndXlsx():
    path_lst1 = [dir2 + '2021_10_22 gisaid_epiflu_sequence_fasta_CutSeqId_8.csv',
                 dir2 + '2022_03_22 gisaid_epiflu_sequence_fasta_CutSeqId_8.csv',
                 dir2 + '2022_10_22 gisaid_epiflu_sequence_fasta_CutSeqId_8.csv']
    path_lst2 = [dir2 + '2021_10_22 gisaid_epiflu_isolates_xls.csv',
                 dir2 + '2022_03_22 gisaid_epiflu_isolates_xls.csv',
                 dir2 + '2022_10_22 gisaid_epiflu_isolates_xls.csv']

    for i in range(len(path_lst1)):
        df = pd.read_csv(path_lst1[i])
        df1 = pd.read_csv(path_lst2[i])
        df2 = pd.DataFrame()
        strain_lst = []
        subtype_lst = []
        host_lst = []
        host0_lst = []
        date_lst = []
        year_lst = []
        country_lst = []
        continent_lst = []
        source_lst = []

        CDS1_lst = []
        CDS1_len_lst = []
        CDS2_lst = []
        CDS2_len_lst = []
        CDS3_lst = []
        CDS3_len_lst = []
        CDS4_lst = []
        CDS4_len_lst = []
        CDS5_lst = []
        CDS5_len_lst = []
        CDS6_lst = []
        CDS6_len_lst = []
        CDS7_lst = []
        CDS7_len_lst = []
        CDS8_lst = []
        CDS8_len_lst = []

        j = 0
        k = 0
        df1['Collection_Date'] = pd.to_datetime(df1['Collection_Date'])
        while k < df.shape[0] and j < df1.shape[0]:
            if df1['Isolate_Id'][j] == df['strain_id'][k] \
                    and df1['PB2 Segment_Id'][j] is not np.nan \
                    and df1['PB1 Segment_Id'][j] is not np.nan \
                    and df1['PA Segment_Id'][j] is not np.nan \
                    and df1['HA Segment_Id'][j] is not np.nan \
                    and df1['NP Segment_Id'][j] is not np.nan \
                    and df1['NA Segment_Id'][j] is not np.nan \
                    and df1['MP Segment_Id'][j] is not np.nan \
                    and df1['NS Segment_Id'][j] is not np.nan \
                    and str(df['unique_1'][k]) in df1['PB2 Segment_Id'][j] \
                    and str(df['unique_2'][k]) in df1['PB1 Segment_Id'][j] \
                    and str(df['unique_3'][k]) in df1['PA Segment_Id'][j] \
                    and str(df['unique_4'][k]) in df1['HA Segment_Id'][j] \
                    and str(df['unique_5'][k]) in df1['NP Segment_Id'][j] \
                    and str(df['unique_6'][k]) in df1['NA Segment_Id'][j] \
                    and str(df['unique_7'][k]) in df1['MP Segment_Id'][j] \
                    and str(df['unique_8'][k]) in df1['NS Segment_Id'][j]:

                strain_lst.append(df1['Isolate_Name'][j])
                date_lst.append(df1['Collection_Date'][j])
                year_lst.append(str(df1['Collection_Date'][j])[:4])
                subtype_lst.append(df['subtype'][k])
                host0_lst.append(df1['Host'][j])
                tmp_lst = str(df1['Location'][j]).split('/')
                continent_lst.append(tmp_lst[0].strip(' '))
                source_lst.append('GISAID')
                if len(tmp_lst) > 1:
                    country_lst.append(tmp_lst[1].strip(' '))
                else:
                    country_lst.append(None)

                if 'Human' in df1['Host'][j]:
                    host_lst.append('Human')
                elif 'Sus' in df1['Host'][j] or 'Swine' in df1['Host'][j] \
                        or 'Pig' in df1['Host'][j]:
                    host_lst.append('Swine')
                else:
                    host_lst.append('Avian')
                CDS1_lst.append(df['CDS_1'][k])
                CDS1_len_lst.append(df['len_CDS_1'][k])
                CDS2_lst.append(df['CDS_2'][k])
                CDS2_len_lst.append(df['len_CDS_2'][k])
                CDS3_lst.append(df['CDS_3'][k])
                CDS3_len_lst.append(df['len_CDS_3'][k])
                CDS4_lst.append(df['CDS_4'][k])
                CDS4_len_lst.append(df['len_CDS_4'][k])
                CDS5_lst.append(df['CDS_5'][k])
                CDS5_len_lst.append(df['len_CDS_5'][k])
                CDS6_lst.append(df['CDS_6'][k])
                CDS6_len_lst.append(df['len_CDS_6'][k])
                CDS7_lst.append(df['CDS_7'][k])
                CDS7_len_lst.append(df['len_CDS_7'][k])
                CDS8_lst.append(df['CDS_8'][k])
                CDS8_len_lst.append(df['len_CDS_8'][k])
                j += 1
                k += 1
            else:
                if df['strain_id'][k] not in list(df1['Isolate_Id']):
                    k += 1
                if df1['Isolate_Id'][j] not in list(df['strain_id']):
                    j += 1

        df2['Strain_name'] = strain_lst
        df2['CDS_1'] = CDS1_lst
        df2['CDS_2'] = CDS2_lst
        df2['CDS_3'] = CDS3_lst
        df2['CDS_4'] = CDS4_lst
        df2['CDS_5'] = CDS5_lst
        df2['CDS_6'] = CDS6_lst
        df2['CDS_7'] = CDS7_lst
        df2['CDS_8'] = CDS8_lst
        df2['CDS_len_1'] = CDS1_len_lst
        df2['CDS_len_2'] = CDS2_len_lst
        df2['CDS_len_3'] = CDS3_len_lst
        df2['CDS_len_4'] = CDS4_len_lst
        df2['CDS_len_5'] = CDS5_len_lst
        df2['CDS_len_6'] = CDS6_len_lst
        df2['CDS_len_7'] = CDS7_len_lst
        df2['CDS_len_8'] = CDS8_len_lst
        df2['Host'] = host_lst
        df2['Host0'] = host0_lst
        df2['Serotype'] = subtype_lst
        df2['Date'] = date_lst
        df2['Year'] = year_lst
        df2['Country'] = country_lst
        df2['Continent'] = continent_lst
        df2['data_label'] = source_lst
        df2.to_csv(dir2 + path_lst2[i].split('/')[3].split('.')[0] + '_8.csv', index=False)


def Merge3GISAID():
    file1 = dir2 + '2021_10_22 gisaid_epiflu_isolates_xls_8.csv'
    file2 = dir2 + '2022_03_22 gisaid_epiflu_isolates_xls_8.csv'
    file3 = dir2 + '2022_10_22 gisaid_epiflu_isolates_xls_8.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.concat([df1, df2, df3], axis=0)

    print(df1.shape)
    print(df2.shape)
    print(df3.shape)
    print(df4.shape)
    df4.to_csv(dir2 + 'AIV_2021_10_22to2023_GISAID_8.csv', index=False)


def Readgb():
    path = './AIV_2021_10_22to2023/NCBI/ncbi_2021_10_22 sequence.gb'

    df = pd.DataFrame()
    strain_lst = []
    subtype_lst = []
    host_lst = []
    segment_lst = []
    date_lst = []
    country_lst = []
    continent_lst = []
    seq_lst = []
    seq_len_lst = []

    for record in SeqIO.parse(path, 'genbank'):
        for feature in record.features:
            feature_type = feature.type
            qualifiers = feature.qualifiers
            flag = np.zeros(6)
            if feature_type == 'source':
                strain = qualifiers.get('strain')
                # print(strain)
                if strain:
                    strain_lst.append(strain[0])
                    flag[0] = 1

                serotype = qualifiers.get('serotype')
                if serotype:
                    subtype_lst.append(serotype[0])
                    flag[1] = 1

                host = qualifiers.get('host')
                if host:
                    host_lst.append(host[0])
                    flag[2] = 1

                segment = qualifiers.get('segment')
                if segment:
                    segment_lst.append(segment[0])
                    flag[3] = 1

                collect_date = qualifiers.get('collection_date')
                if collect_date:
                    date_lst.append(collect_date[0])
                    flag[4] = 1

                country = qualifiers.get('country')
                if country:
                    country_lst.append(country[0])
                    flag[5] = 1

                if 0 not in flag:
                    location = feature.location
                    seq = location.extract(record.seq)
                    seq_lst.append(seq)
                    seq_len_lst.append(len(seq))
                else:
                    if len(strain_lst) != 0 and flag[0] == 1:
                        strain_lst.pop()
                    if len(subtype_lst) != 0 and flag[1] == 1:
                        subtype_lst.pop()
                    if len(host_lst) != 0 and flag[2] == 1:
                        host_lst.pop()
                    if len(segment_lst) != 0 and flag[3] == 1:
                        segment_lst.pop()
                    if len(date_lst) != 0 and flag[4] == 1:
                        date_lst.pop()
                    if len(country_lst) != 0 and flag[5] == 1:
                        country_lst.pop()
                flag = np.zeros(6)

    df['strain'] = strain_lst
    df['seq'] = seq_lst
    df['seq_len'] = seq_len_lst
    df['segment'] = segment_lst
    df['host'] = host_lst
    df['subtype'] = subtype_lst
    df['date'] = date_lst
    df['country'] = country_lst
    data = df.sort_values(by=['strain', 'segment'], ascending=True)
    data.to_csv(dir3 + "AIV_2021_10_22to2023_NCBI_gb_CutSeqId.csv", index=False)


def TurnContinent(country):
    continent_dic = {1: 'Asia', 2: 'Europe', 3: 'North America', 4: 'South America',
                     5: 'Africa', 6: 'Oceania', 7: 'Antarctica'}
    asia_lst = ['Iran', 'Thailand', 'Myanmar', 'Kazakhstan', 'Singapore', 'Japan', 'China', 'Mongolia',
                'Bangladesh', 'Pakistan', 'Cambodia', 'Hong Kong', 'Viet Nam', 'India', 'South Korea']
    europe_lst = ['Germany', 'Italy', 'Belgium', 'Russia', 'Iceland', 'United Kingdom', 'Netherlands',
                  'France', 'Czech Republic', 'Denmark', 'Spain']
    northamr_lst = ['USA', 'Mexico', 'Canada', 'El Salvador', 'Guatemala']
    southamr_lst = ['Chile', 'Brazil', 'Venezuela', 'Peru', 'Argentina']
    africa_lst = ['Nigeria', 'Lesotho', 'Benin', 'Uganda', 'Namibia', 'South Africa', 'Egypt']
    oceania_lst = ['Australia']

    if country in asia_lst:
        return continent_dic[1]
    elif country in europe_lst:
        return continent_dic[2]
    elif country in northamr_lst:
        return continent_dic[3]
    elif country in southamr_lst:
        return continent_dic[4]
    elif country in africa_lst:
        return continent_dic[5]
    elif country in oceania_lst:
        return continent_dic[6]


def MergeNCBI8Seg():
    path = dir3 + 'AIV_2021_10_22to2023_NCBI_gb_CutSeqId.csv'
    df = pd.read_csv(path)

    df1 = pd.DataFrame()
    strain_lst = []
    subtype_lst = []
    host_lst = []
    host0_lst = []
    date_lst = []
    year_lst = []
    country_lst = []
    continent_lst = []
    source_lst = []

    CDS1_lst = []
    CDS1_len_lst = []
    CDS2_lst = []
    CDS2_len_lst = []
    CDS3_lst = []
    CDS3_len_lst = []
    CDS4_lst = []
    CDS4_len_lst = []
    CDS5_lst = []
    CDS5_len_lst = []
    CDS6_lst = []
    CDS6_len_lst = []
    CDS7_lst = []
    CDS7_len_lst = []
    CDS8_lst = []
    CDS8_len_lst = []

    flag = np.zeros(8)
    i = 0
    strain = ''
    df['date'] = pd.to_datetime(df['date'])
    while i < df.shape[0]:
        if 0 not in flag:
            if df['segment'][i] == 1:
                flag = np.zeros(8)
                i -= 1
        elif flag[0] == 0 and df['segment'][i] == 1:
            strain = df['strain'][i]
            strain_lst.append(df['strain'][i])
            subtype_lst.append(df['subtype'][i])
            host0_lst.append(df['host'][i])
            country_lst.append(str(df['country'][i]).split(':')[0])
            continent_lst.append(TurnContinent(str(df['country'][i]).split(':')[0]))
            source_lst.append('NCBI')
            if 'Homo sapiens' in df['host'][i]:
                host_lst.append('Human')
            elif 'Sus' in df['host'][i] or 'swine' in df['host'][i]:
                host_lst.append('Swine')
            else:
                host_lst.append('Avian')
            date_lst.append(df['date'][i])
            year_lst.append(str(df['date'][i])[:4])
            flag[0] = 1
            CDS1_lst.append(df['seq'][i])
            CDS1_len_lst.append(df['seq_len'][i])
        elif flag[1] == 0 and df['segment'][i] == 2:
            flag[1] = 1
            CDS2_lst.append(df['seq'][i])
            CDS2_len_lst.append(df['seq_len'][i])
        elif flag[2] == 0 and df['segment'][i] == 3:
            flag[2] = 1
            CDS3_lst.append(df['seq'][i])
            CDS3_len_lst.append(df['seq_len'][i])
        elif flag[3] == 0 and df['segment'][i] == 4:
            flag[3] = 1
            CDS4_lst.append(df['seq'][i])
            CDS4_len_lst.append(df['seq_len'][i])
        elif flag[4] == 0 and df['segment'][i] == 5:
            flag[4] = 1
            CDS5_lst.append(df['seq'][i])
            CDS5_len_lst.append(df['seq_len'][i])
        elif flag[5] == 0 and df['segment'][i] == 6:
            flag[5] = 1
            CDS6_lst.append(df['seq'][i])
            CDS6_len_lst.append(df['seq_len'][i])
        elif flag[6] == 0 and df['segment'][i] == 7:
            flag[6] = 1
            CDS7_lst.append(df['seq'][i])
            CDS7_len_lst.append(df['seq_len'][i])
        elif flag[7] == 0 and df['segment'][i] == 8:
            flag[7] = 1
            CDS8_lst.append(df['seq'][i])
            CDS8_len_lst.append(df['seq_len'][i])
        i += 1

    df1['Strain_name'] = strain_lst
    df1['CDS_1'] = CDS1_lst
    df1['CDS_2'] = CDS2_lst
    df1['CDS_3'] = CDS3_lst
    df1['CDS_4'] = CDS4_lst
    df1['CDS_5'] = CDS5_lst
    df1['CDS_6'] = CDS6_lst
    df1['CDS_7'] = CDS7_lst
    df1['CDS_8'] = CDS8_lst
    df1['CDS_len_1'] = CDS1_len_lst
    df1['CDS_len_2'] = CDS2_len_lst
    df1['CDS_len_3'] = CDS3_len_lst
    df1['CDS_len_4'] = CDS4_len_lst
    df1['CDS_len_5'] = CDS5_len_lst
    df1['CDS_len_6'] = CDS6_len_lst
    df1['CDS_len_7'] = CDS7_len_lst
    df1['CDS_len_8'] = CDS8_len_lst
    df1['Host'] = host_lst
    df1['Host0'] = host0_lst
    df1['Serotype'] = subtype_lst
    df1['Date'] = date_lst
    df1['Year'] = year_lst
    df1['Country'] = country_lst
    df1['Continent'] = continent_lst
    df1['data_label'] = source_lst
    df1.to_csv(dir3 + 'AIV_2021_10_22to2023_NCBI_8.csv', index=False)


def MergeAndDuplicate():
    file1 = dir2 + 'AIV_2021_10_22to2023_GISAID_8.csv'
    file2 = dir1 + 'AIV_2021_10_22to2023_BVBRC_fasta_xlsx_8.csv'
    file3 = dir3 + 'AIV_2021_10_22to2023_NCBI_8.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.concat([df1, df2, df3], axis=0)

    print(df4.shape)
    df4.drop_duplicates(subset=['Strain_name'], keep='first', inplace=True)
    print(df4.shape)
    df4.to_csv(dir4 + 'AIV_2021_10_22to2023_8_' + str(df4.shape[0]) + '.csv', index=False)


def Merge2File():
    file1 = dir4 + 'df0_IAVs_8segs_unique_concat_intersection_NCBI,GISAID,IRD.csv'
    file2 = dir4 + 'AIV_2021_10_22to2023_8_34139.csv'
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.concat([df1, df2], axis=0)
    print(df3.shape)
    df3.drop_duplicates(subset=['Strain_name'], keep='first', inplace=True)
    df4 = df3[(df3['CDS_len_1'] >= 2020) & (df3['CDS_len_2'] >= 2050) & (df3['CDS_len_3'] >= 1920)
              & (df3['CDS_len_4'] >= 1650) & (df3['CDS_len_5'] >= 1480) & (df3['CDS_len_6'] >= 1330)]
    print(df3.shape)
    print(df4.shape)
    df4.to_csv(dir4 + 'AIV_all_8_' + str(df4.shape[0]) + '.csv', index=False)


file_lst = ['pb2', 'pb1', 'pa', 'np']


def CsvToFasta():
    file = dir4 + 'AIV_all_8_79630.csv'
    df = pd.read_csv(file)
    nt_lst = ['A', 'T', 'C', 'G', 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D', 'N']
    cds_lst_name = ['CDS_1', 'CDS_2', 'CDS_3', 'CDS_5']
    CDS_lst = [[] for _ in range(4)]
    for i in range(df.shape[0]):
        for j in range(4):
            str = df[cds_lst_name[j]][i].upper()
            tmp_str = ''
            for k in str:
                if k not in nt_lst:
                    tmp_str += '-'
                else:
                    tmp_str += k
            CDS_lst[j].append(tmp_str)

    for i in range(4):
        df[cds_lst_name[i]] = CDS_lst[i]

    for j in range(4):
        fp = open(dir5 + file_lst[j] + '.fasta', 'w')
        for i in range(df.shape[0]):
            print('>' + df['Strain_name'][i] + '\n' + df[cds_lst_name[j]][i], file=fp)
        fp.close()


if __name__ == '__main__':
    # BVBRC
    CutFasta()
    CutBVBRCSeqId()
    ExtractBVBRCXlsx()
    CombineBVBRCFastaAndXlsx()
    MergeBVBRC8Seg()

    # GISAID
    CutGISAIDId()
    ExtractGISAIDXlsx()
    MergeGISAID8Seg()
    MergeGISAIDFastaAndXlsx()
    Merge3GISAID()

    # NCBI
    Readgb()
    MergeNCBI8Seg()

    # merge
    MergeAndDuplicate()
    Merge2File()

    # Generate a FASTA file and then utilize mafft for sequence alignment
    CsvToFasta()
