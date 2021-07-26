import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from ers.similarity_combination_element_lib import SimilarityCombinationElement
from ers.post_processing import generate_linkage_matrix, sort_matrix

def ax_setting():
    plt.style.use('default')
    plt.tick_params(axis='x', which='major', labelsize=24)
    plt.tick_params(axis='y', which='major', labelsize=24)

def main():
    # Learn similarity matrix
    df_data = pd.read_csv(INPUT_DATA, index_col=0)
    similarity_combination_element = SimilarityCombinationElement(
        df_data=df_data, similarity_value=SIMILARITY_VALUE, rage_size_subset=MAX_SIZE_SUBSET
    )
    similarity_combination_element.similarity_measurement()
    df_similarity = similarity_combination_element.df_similarity
    df_dissimilarity = similarity_combination_element.df_dissimilarity
    df_uncertainty = similarity_combination_element.df_uncertainty
    
    # Use the hierarchical clustering to group elements
    linkage_matrix, distance_matrix = generate_linkage_matrix(
        df_similarity.loc[ELEMENTS,ELEMENTS], df_dissimilarity.loc[ELEMENTS,ELEMENTS], df_uncertainty.loc[ELEMENTS,ELEMENTS], 
        method=METHOD, metric=METRIC
    )
    
    # Visualize the dendogram
    dendogram_figure = plt.figure(figsize=(16, 16))

    dendrogram = hierarchy.dendrogram(
        Z=linkage_matrix, 
        labels=ELEMENTS,
        leaf_font_size=20
    )

    ax_setting()
    plt.xticks(rotation=90)
    plt.savefig("{}/dendogram.pdf".format(OUTPUT_FOLDER), dpi=300, bbox_inches='tight')
    
    # Visualize the similarity matrix sorted using the order of elements in dendogram
    plt.figure(figsize=(10,8), dpi=300)
    ax_setting()
    df_matrix = sort_matrix(df_similarity.loc[ELEMENTS, ELEMENTS], order=dendrogram["ivl"]) # Sort the order of elements in similarity matrix using the order obtained from dendogram
    ax = sns.heatmap(df_matrix, cmap="YlGnBu", xticklabels=df_matrix.columns.values, 
        yticklabels=df_matrix.index.values, vmax=1, vmin=0
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.savefig("{}/similarity.pdf".format(OUTPUT_FOLDER), dpi=300, bbox_inches='tight')

INPUT_DATA = "../data/HEA_data.demo.csv"
OUTPUT_FOLDER = "../results/demo_1"
SIMILARITY_VALUE = 0.1 # parameter alpha
MAX_SIZE_SUBSET = 2 # max size of elements combinations measured the similarity and used to predict the property of alloy
ELEMENTS = ['Ag', 'Au', 'Co', 'Cr', 'Cu', 'Fe', 'Mo', 'Ni', 'Pd', 'Pt']
METHOD = "ward"
METRIC = "euclidean"

if __name__ == '__main__':
    main()
    