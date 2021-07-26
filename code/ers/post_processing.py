import pandas as pd 
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

def convert_dst_to_distance_matrix(df_similarity_matrix, df_dissimilarity_matrix, df_unknown_matrix):
	"""
	Return distance matrix between (1) elements, (2) element and combination of element, and (3) combinations of element.
	The distance matrix is converted from the similarity matrix
	"""
	max_smt = np.max(df_similarity_matrix.values)
	tmp_similarity_matrix = np.copy(df_similarity_matrix.values)
	distance_matrix = np.power(1 - tmp_similarity_matrix, 1/2)
	np.fill_diagonal(distance_matrix, 0)
	return distance_matrix

def generate_linkage_matrix(
	df_similarity_matrix, df_dissimilarity_matrix, df_unknown_matrix,
	method="ward", metric="euclidean"
):
	"""
	Return linkage matrix calculated based on the obtained similarity matrix.

	The linkage matrix is used to build a hierarchical structure between (1) elements, (2) element and combination of element, and (3) combinations of element.
	The function based on the scipy.linkage module

	method: The linkage algorithm to use. See the Linkage Methods section below for full descriptions.
	metric: The distance metric to use in the case that y is a collection of observation vectors; ignored otherwise. See the pdist function for a list of valid distance metrics. A custom distance function can also be used.
	"""
	distance_matrix = convert_dst_to_distance_matrix(df_similarity_matrix, df_dissimilarity_matrix, df_unknown_matrix)
	np.fill_diagonal(distance_matrix, 0)
	dist_condens = squareform(distance_matrix)

	linkage_matrix = hierarchy.linkage(dist_condens, method=method, metric=metric)

	return linkage_matrix, distance_matrix

def sort_matrix(df_matrix, order):
	"""
	Return sorted matrix.

	The function sorts a paiwise similarity matrix based on the order of instances in the matrix.
	"""
	tmp_data = []
	for label in order:
		tmp_data.append(df_matrix.loc[label])
	optimize_df = pd.DataFrame(tmp_data)
	optimize_df = optimize_df.reindex(order, axis=1)
	values_matrix = optimize_df.values
	features = optimize_df.columns.values
	values_matrix = np.rot90(values_matrix)
	df_result = pd.DataFrame(values_matrix, columns=features)
	df_result.index = features[::-1]
	return df_result