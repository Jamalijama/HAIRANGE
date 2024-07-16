import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# load the data

file_name = 'before2020_29634_np'
matrix_1 = np.load('../Res/npy/array_codon_freq_' + file_name + '.npy', allow_pickle=True)
label = np.load('../Res/npy/array_label_' + file_name + '.npy', allow_pickle=True)
matrix_2 = np.load('../Res/npy_trm/array_codon_freq_' + file_name + '_trm.npy', allow_pickle=True)

matrix_1 = matrix_1.reshape(matrix_1.shape[0], -1)
matrix_2 = matrix_2.reshape(matrix_2.shape[0], -1)

# Reduce the dimensionality of the data
umap = umap.UMAP(n_neighbors=100, n_components=2, random_state=7, min_dist=0.5)
pca = PCA(n_components=2, random_state=7)
tsne = TSNE(n_components=2, random_state=7)

matrix_1_pca = pca.fit_transform(matrix_1)
matrix_2_pca = pca.fit_transform(matrix_2)

matrix_1_tsne = tsne.fit_transform(matrix_1)
matrix_2_tsne = tsne.fit_transform(matrix_2)

matrix_1_umap = umap.fit_transform(matrix_1, label)
matrix_2_umap = umap.fit_transform(matrix_2, label)

df_1 = pd.DataFrame()
df_1['pca1'] = matrix_1_pca[:, 0]
df_1['pca2'] = matrix_1_pca[:, 1]
df_1['tsne1'] = matrix_1_tsne[:, 0]
df_1['tsne2'] = matrix_1_tsne[:, 1]
df_1['umap1'] = matrix_1_umap[:, 0]
df_1['umap2'] = matrix_1_umap[:, 1]
df_1['label'] = label
df_1.to_csv('../Res/trm_feature_csv/features_before_trm_' + file_name + '_100+0.5.csv', index=False)

df_2 = pd.DataFrame()
df_2['pca1'] = matrix_2_pca[:, 0]
df_2['pca2'] = matrix_2_pca[:, 1]
df_2['tsne1'] = matrix_2_tsne[:, 0]
df_2['tsne2'] = matrix_2_tsne[:, 1]
df_2['umap1'] = matrix_2_umap[:, 0]
df_2['umap2'] = matrix_2_umap[:, 1]
df_2['label'] = label
df_2.to_csv('../Res/trm_feature_csv/features_after_trm_' + file_name + '_100+0.5.csv', index=False)
