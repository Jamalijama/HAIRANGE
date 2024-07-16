import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score, accuracy_score

umap = umap.UMAP(n_neighbors=100, n_components=2, random_state=7, min_dist=0.5)
pca = PCA(n_components=2, random_state=7)
tsne = TSNE(n_components=2, random_state=7)

name_lst = []
before_trm_lst = []
after_trm_lst = []
label_lst = []


def eval(file_name, dir1, file):
    print(state, num)
    name_lst.append(str(state) + '+' + str(num))
    before_trm_lst.append('x')
    after_trm_lst.append('x')
    label_lst.append('x')

    df = pd.read_excel(dir1 + file)
    df_avian = df[df['Host_IRD'] == 0]
    df_human = df[df['Host_IRD'] == 1]
    df_human_sample = df_human.sample(n=num, random_state=state)
    df_avian_sample = df_avian.sample(n=num, random_state=state)

    index_lst = sorted(df_human_sample.index.tolist() + df_avian_sample.index.tolist())

    num1 = len(index_lst)

    matrix_11 = np.load('../Res/npy/array_codon_freq_' + file_name + '.npy', allow_pickle=True)
    matrix_22 = np.load('../Res/npy_trm/array_codon_freq_' + file_name + '_trm.npy', allow_pickle=True)
    label_11 = np.load('../Res/npy/array_label_' + file_name + '.npy', allow_pickle=True)

    matrix_1 = matrix_11[index_lst]
    matrix_2 = matrix_22[index_lst]
    label_1 = label_2 = label_11[index_lst]
    print('before:', matrix_1.shape)

    matrix_1 = matrix_1.reshape(matrix_1.shape[0], -1)
    matrix_2 = matrix_2.reshape(matrix_2.shape[0], -1)
    print('after:', matrix_1.shape)

    for i in range(1):
        # dimension reduction
        matrix_1_umap = umap.fit_transform(matrix_1, label_1)
        matrix_2_umap = umap.fit_transform(matrix_2, label_2)

        # clustering
        ac_cluster_1_pred = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(matrix_1_umap)
        ac_cluster_2_pred = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(matrix_2_umap)

        # 聚类效果评价
        print('\nagglomerative clustering:')
        # accuracy，值越大越好
        acc_1 = accuracy_score(label_1, ac_cluster_1_pred)
        acc_2 = accuracy_score(label_2, ac_cluster_2_pred)
        while acc_1 < 0.5 or acc_2 < 0.5:
            if acc_1 < 0.5:
                ac_cluster_1_pred = [1 - x for x in ac_cluster_1_pred]
            if acc_2 < 0.5:
                ac_cluster_2_pred = [1 - x for x in ac_cluster_2_pred]
            acc_1 = accuracy_score(label_1, ac_cluster_1_pred)
            acc_2 = accuracy_score(label_2, ac_cluster_2_pred)
        print('acc:', acc_1, acc_2)
        name_lst.append('agglomerative clustering+acc')
        before_trm_lst.append(acc_1)
        after_trm_lst.append(acc_2)
        if acc_1 < acc_2:
            label_lst.append(1)
        else:
            label_lst.append(0)

        # 调整兰德指数，值越大越好
        ari_score_1 = adjusted_rand_score(ac_cluster_1_pred, label_1)
        ari_score_2 = adjusted_rand_score(ac_cluster_2_pred, label_2)
        print("ari_score:", ari_score_1, ari_score_2)
        name_lst.append('agglomerative clustering+ari_score')
        before_trm_lst.append(ari_score_1)
        after_trm_lst.append(ari_score_2)
        if ari_score_1 <= ari_score_2:
            label_lst.append(1)
        else:
            label_lst.append(0)

        # 标准化互信息，值越大越好
        nmi_score_1 = normalized_mutual_info_score(ac_cluster_1_pred, label_1)
        nmi_score_2 = normalized_mutual_info_score(ac_cluster_2_pred, label_2)
        print('nmi_score:', nmi_score_1, nmi_score_2)
        name_lst.append('agglomerative clustering+nmi_score')
        before_trm_lst.append(nmi_score_1)
        after_trm_lst.append(nmi_score_2)
        if nmi_score_1 <= nmi_score_2:
            label_lst.append(1)
        else:
            label_lst.append(0)


state_lst = [7, 11, 14]
num_lst = [1500]

file_name = 'before2020_29634_np'

dir1 = '../DataCleaning/Res/res1/'
file = 'AIV_all_8_before2020_29634.xlsx'

for state in state_lst:
    for num in num_lst:
        eval(file_name, dir1, file)
df = pd.DataFrame()
df['name'] = name_lst
df['before_transfomer'] = before_trm_lst
df['after_transformer'] = after_trm_lst
df['label'] = label_lst
df.to_csv('../Res/pre_processing_evaluate/result_' + file_name + '_umap_agcluster.csv')
