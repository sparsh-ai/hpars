import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from ers.similarity_combination_element_lib import SimilarityCombinationElement, InstanceBasedClassifier

def main():
    # Load data set
    df_data = pd.read_csv(INPUT_DATA, index_col=0)
    
    # Learn similarity matrix
    similarity_combination_element = SimilarityCombinationElement(
        df_data=df_data, similarity_value=SIMILARITY_VALUE, rage_size_subset=MAX_SIZE_SUBSET
    )
    
    similarity_combination_element.similarity_measurement()
    
    df_similarity = similarity_combination_element.df_similarity
    df_dissimilarity = similarity_combination_element.df_dissimilarity
    df_uncertainty = similarity_combination_element.df_uncertainty

    # Estimate the belief of properties of alloys
    classifier = InstanceBasedClassifier(
        df_data, df_similarity, df_dissimilarity, 
        df_uncertainty, n_gram_evidence=MAX_SIZE_SUBSET
    )
    
    y_pred = classifier.predict(df_data["set_name"].values)
    print(classification_report(df_data["Label"].values, y_pred))

INPUT_DATA = "../data/HEA_data.demo.csv"
SIMILARITY_VALUE = 0.1 # parameter alpha
MAX_SIZE_SUBSET = 2 # max size of elements combinations measured the similarity and used to predict the property of alloy

if __name__ == '__main__':
    main()
    