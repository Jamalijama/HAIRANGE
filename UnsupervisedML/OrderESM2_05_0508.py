import numpy as np
import pandas as pd


cds_lst = ['pb2', 'pb1', 'pa', 'np']
len_lst = [768, 768, 768, 512]

for i in range(len(cds_lst)):
# for i in range(1,2):
    cds = cds_lst[i]
    res_name = 'ha_indp_' + cds
    max_len = len_lst[i]
    
    matrixs = np.load('../Res/model_evaluate/ESM2_05/new_AIV_all_8_' + res_name + '_protein_emb_esm2_array_0503.npy', allow_pickle=True)
    ids = np.load('../Res/model_evaluate/ESM2_05/new_AIV_all_8_' + res_name + '_protein_emb_esm2_id_0503.npy', allow_pickle=True)
    im_dict = {}
    for i in range(ids.shape[0]):
        if matrixs[i].shape[0] < max_len:
            matrix_lst = list(matrixs[i])
            for j in range(max_len - matrixs[i].shape[0]):
                matrix_lst.append([0 for _ in range(2560)])
                # print(ids[i].split('_')[1])
                # print(type(ids[i]))
        im_dict[str(ids[i])] = matrix_lst
        
    # print(im_dict)
    
    matrix_new = []
    res_name1 = 'indp_' + cds
    df = pd.read_csv('../DataCleaning/Res/res1/IAV_reassortant_1450to2500_new_4_' + res_name1 + '.csv')
    id_lst = df['id'].tolist()
    for i in range(df.shape[0]):
        idd = id_lst[i].replace(' ', '').replace('/', '_').split('$')[0]
        # idd = id_lst[i].replace('/', '$')
        matrix_new.append(im_dict['matrix_' + idd])
    
    array_new = np.array(matrix_new)
    np.save('../Res/model_evaluate/ESM2_05/new_AIV_all_8_' + res_name + '_protein_emb_esm2_array_0503_ordered.npy', array_new, allow_pickle=True)

#matrixs = np.load('../Res/model_evaluate/ESM2/new_AIV_all_8_ha_sampled1000_pb1_protein_emb_esm2_array_ordered.npy', allow_pickle=True)
#
#print(matrixs.shape)