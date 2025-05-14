from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# from imblearn.over_sampling import SMOTE
amino_table = ['I', 'D', 'M', 'H', 'E', 'W', 'R', 'L', 'Y', 'Q', 'G', 'A', 'S', 'P', 'C', 'T', 'V', 'F', 'N', 'K']
nt_table = ['t', 'c', 'a', 'g']
dnt_table = [nt1 + nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1 + dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
codon_table = [nt1 + nt2 + nt3 for nt1 in nt_table for nt2 in nt_table for nt3 in nt_table]
codon_table1 = codon_table.copy()
codon_table1.remove('taa')
codon_table1.remove('tag')
codon_table1.remove('tga')
codonpair_table = [codon0 + codon1 for codon0 in codon_table1 for codon1 in codon_table1]

dnt_category = ['n12', 'n23', 'n31']
dntpair_category = ['n12m12', 'n23m23', 'n31m31', 'n12n31', 'n23m12', 'n31m23']

dnts_cols_list = ['Freq_' + dnt + '_' + dnt_cat for dnt_cat in dnt_category for dnt in dnt_table]
dntpair_cols_list = ['Freq_' + dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]
codon_cols_list = ['Freq_' + codon for codon in codon_table]
codonpair_cols_list = ['Freq_' + codonpair for codonpair in codonpair_table]
amino_cols_list = ['Freq_' + amino for amino in amino_table]

full_cols_list0 = dnts_cols_list + dntpair_cols_list + codon_cols_list + codonpair_cols_list + amino_cols_list
full_cols_list1 = [col[5:] for col in full_cols_list0]

cnames = {
    'lightblue': '#ADD8E6',
    'deepskyblue': '#00BFFF',
    'cadetblue': '#5F9EA0',
    'cyan': '#00FFFF',
    'purple': '#800080',
    'orchid': '#DA70D6',
    'lightgreen': '#90EE90',
    'darkgreen': '#006400',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32',
    'deeppink': '#FF1493',
    'burlywood': '#DEB887',
    'red': '#FF0000',
    'indianred': '#CD5C5C',
    'darkred': '#8B0000',
}
# cnames = {
# 'darkgreen':            '#006400',
# # 'lightblue':            '#ADD8E6',
# 'yellow':               '#FFFF00',
# 'yellowgreen':          '#9ACD32',
# 'burlywood':            '#DEB887',
# 'blue1':                '#1663A9',
# 'gray':                 '#666666',
# 'darkgreen':            '#006400',
# 'darkyellow':           '#996600',
# 'purple':               '#800080',
# # 'darkred':              '#8B0000',
# 'deeppink':             '#FF1493',
# 'green':                '#66CC00',
# 'deepskyblue':          '#00BFFF',
# 'orange1':              '#FF6600',
# 'orchid':               '#DA70D6',
# 'yellow':               '#FFCC33',
# 'blue2':                '#6EB1DE',
# 'lightgreen':           '#90EE90',
# # 'blue3':                '#1383C2',
# # 'blue4':                '#20C2L1',
# # 'deepskyblue':          '#00BFFF',
#     }

color_num_list = list(range(1, 16, 1))

# print (len(color_num_list))
color_dict = dict(zip(color_num_list, cnames.values()))
# print (color_dict)
color_list0 = list(color_dict.values())

cds_lst = ['CDS_1', 'CDS_2', 'CDS_3', 'CDS_5']
#for i in range(len(cds_lst)):
for i in range(0,1):
    cds = cds_lst[i]
    res_name = 'ha_sampled1000_seed17_' + cds
    df1 = pd.read_excel('../DataCleaning/Res/res1/df_full_DCR_counting_new_' + res_name + '.xlsx')
    matrix_npy = np.array(df1[dntpair_cols_list])
    np.save('../Res/model_evaluate/DCR/DCR_' + res_name + '.npy', matrix_npy, allow_pickle=True)
    print(matrix_npy.shape)
    label_1 = list(df1['label'])

    y_types = sorted(set(label_1))
    y_num = len(y_types)

    reshape_composition_lst = []

    for i in range(len(matrix_npy)):
        sample_composition = matrix_npy[i]
        reshape_composition0 = np.reshape(sample_composition, (1, 1536))
        reshape_composition = reshape_composition0[0]
        reshape_composition_lst.append(reshape_composition)
        # print(reshape_composition)

    reshape_composition_array = np.array(reshape_composition_lst)
    # X_tsne = TSNE(learning_rate=100).fit_transform(reshape_composition_array)
    X_pca = PCA(n_components=2).fit_transform(reshape_composition_array)
    k_selected = len(y_types)
    ac_cluster_1_pred = MiniBatchKMeans(n_clusters=k_selected, random_state=10).fit_predict(X_pca)

    # ac_cluster_1_pred = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(X_pca)
    # # 聚类效果评价
    # print('\nagglomerative clustering:')
    # # accuracy，值越大越好
    # acc_1 = accuracy_score(label_1, ac_cluster_1_pred)
    # print('accuracy=',acc_1)
    # # 轮廓系数，值越大越好
    sh_score_1 = silhouette_score(X_pca, ac_cluster_1_pred, metric='euclidean')
    print('sh_score_1=', sh_score_1)
    # # Calinski-Harabaz Index，值越大越好
    ch_score_1 = calinski_harabasz_score(X_pca, ac_cluster_1_pred)
    print('ch_score_1=', ch_score_1)
    # #  Davies-Bouldin Index(戴维森堡丁指数)，值越小越好
    dbi_score_1 = davies_bouldin_score(X_pca, ac_cluster_1_pred)
    print('dbi_score_1=', dbi_score_1)
    # # 调整兰德指数，值越大越好
    ari_score_1 = adjusted_rand_score(label_1, ac_cluster_1_pred)
    print('ari_score_1=', ari_score_1)
    # # 标准化互信息，值越大越好
    nmi_score_1 = normalized_mutual_info_score(label_1, ac_cluster_1_pred)
    print('nmi_score_1=', nmi_score_1)
    #
    # acc_1_lst = []
    sh_score_1_lst = []
    ch_score_1_lst = []
    dbi_score_1_lst = []
    ari_score_1_lst = []
    nmi_score_1_lst = []
    #
    # acc_1_lst.append(acc_1)
    sh_score_1_lst.append(sh_score_1)
    ch_score_1_lst.append(ch_score_1)
    dbi_score_1_lst.append(dbi_score_1)
    ari_score_1_lst.append(ari_score_1)
    nmi_score_1_lst.append(nmi_score_1)
    #
    df_evaluation = pd.DataFrame()
    # df_evaluation['accuracy_score'] = acc_1_lst
    df_evaluation['silhouette_score'] = sh_score_1_lst
    df_evaluation['calinski_harabasz_score'] = ch_score_1_lst
    df_evaluation['davies_bouldin_score'] = dbi_score_1_lst
    df_evaluation['adjusted_rand_score'] = ari_score_1_lst
    df_evaluation['normalized_mutual_info_score'] = nmi_score_1_lst
    #
    df_evaluation.to_csv('../Res/model_evaluate/DCR/df_evaluate_cluster_' + res_name + '_PCA.csv')

    # df_tsne = pd.DataFrame (X_tsne,columns = ['t_SNE1','t_SNE2'])
    # df_tsne = (df_tsne - df_tsne.min()) / (df_tsne.max() - df_tsne.min())
    # df_tsne ['label'] = host_name_array.tolist()

    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())
    df_pca['label'] = label_1
    #
    df1['PCA1'] = df_pca['PCA1']
    df1['PCA2'] = df_pca['PCA2']
    df1['MiniBatchKMeans_label'] = ac_cluster_1_pred
    df1.to_csv('../Res/model_evaluate/DCR/PCA_result_with_MiniBatchKMeans_label_' + res_name + '.csv')

    # plt.figure(figsize=(8, 3))
    # plt.subplot(121)
    # sns.scatterplot(data = df_tsne, x = 't_SNE2', y = 't_SNE1', hue = 'label', palette = color_list0[:y_num],hue_order = y_types) #
    # plt.savefig('sns_scatterplot_tSNE_' + method_name + '.png', dpi = 300, bbox_inches = 'tight')

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    sns.set(font_scale=0.3)
    sns.set_style('white')
    sns.scatterplot(data=df_pca, x='PCA2', y='PCA1', hue='label', palette=color_list0[:y_num], hue_order=y_types)

    plt.savefig('../Res/model_evaluate/DCR/sns_scatterplot_PCA_' + res_name + '.png', dpi=300, bbox_inches='tight')
