import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from ers.similarity_combination_element_lib import SimilarityCombinationElement, InstanceBasedClassifier

def generate_unobserved_combinations(df_data, elements, sizes=[2,3]):
    unobserved_combinations = []
    for size in sizes:
        for unobserved_combination in itertools.combinations(elements, size):
            query = " == 1 & ".join(unobserved_combination) + " == 1 & length == {}".format(len(unobserved_combination))
            df_tmp = df_data.query(query)
            if len(df_tmp.index) == 0:
                unobserved_combinations.append("|".join(unobserved_combination))
    return unobserved_combinations

def report_recommendation_results(unobserved_combinations, final_decisions, df_test):
    # Ranking unobserved combination using the m({HEA})
    results = []
    for material, final_decision in zip(unobserved_combinations, final_decisions):
        results.append([
            material, final_decision[frozenset({"High"})], final_decision[frozenset({"Low"})],
            final_decision[frozenset({"High", "Low"})], "High" if final_decision[frozenset({"High"})] > final_decision[frozenset({"Low"})] else "Low"
        ])

    df_recommend = pd.DataFrame(results, columns=["materials", "DST_High", "DST_Low", "DST_Uncertainty", "Label"])
    df_ranking = df_recommend.sort_values(by=["DST_High", "DST_Uncertainty"], ascending=[False, True])

    # Save the recommendation results of HEA in test set
    results = []
    index = 1
    for ith, row in df_ranking.iterrows():
        elements = row["materials"].split("|")
        df_tmp = df_test.query(" == 1 & ".join(elements) + " == 1 & length == {}".format(len(elements)))
        if len(df_tmp.index) == 1:
            tmp_row = list(row.values.copy())
            tmp_row.append(index)
            results.append(tmp_row)
        index += 1
    return pd.DataFrame(results, columns=["materials", "DST_High", "DST_Low", "DST_Uncertainty", "Label", "index"])

def calculate_mean_ranking_indices(df_data):
    ranking_results = {}
    for i in range(N_SPLITS*N_REPEATS):
        df_recommend_file = pd.read_csv("{}/high_recommend_model_{}.csv".format(OUTPUT_FOLDER, i))
        for ith, row in df_recommend_file.iterrows():
            material = row["materials"]
            atoms = material.split("|")
            df_tmp = df_data.query(" == 1 & ".join(atoms) + " == 1 & length == {}".format(len(atoms)))
            if df_tmp["Label"].values[0] == "High" and len(df_tmp.index) == 1:
                if material in ranking_results.keys():
                    ranking_results[material].append(row["index"])
                else:
                    ranking_results[material] = [row["index"]]
    mean_ranking_HEA = np.mean(np.array(list(ranking_results.values())), axis=1)
    return mean_ranking_HEA

def visualize_ranking_indices(mean_ranking_HEA):
    plt.figure(figsize=(8,8))
    sns.kdeplot(np.log10(mean_ranking_HEA), shade=True, color="red", linewidth=2)
    plt.xlim([0, np.log10(N_CANDIDATE)])
    plt.style.use('default')
    plt.xlabel("Logarithm of rank", {'fontname': 'serif', 'size': 20})
    plt.ylabel("Density", {'fontname': 'serif', 'size': 20})
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.savefig("{}/recommendation_cv.pdf".format(OUTPUT_FOLDER), dpi=300, bbox_inches='tight')

def main():
    # Load data set
    df_data = pd.read_csv(INPUT_DATA, index_col=0)
    
    # Evaluate the recommendation performance of the ERS using cross-validation
    kf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=1)
    for index_model, (train, test) in enumerate(kf.split(df_data.index.values)):
        print("==========CV {}==========".format(index_model))
        df_data_train = df_data.loc[train]
        df_data_test = df_data.loc[test]
        
        similarity_combination_element = SimilarityCombinationElement(
            df_data=df_data, similarity_value=SIMILARITY_VALUE, rage_size_subset=MAX_SIZE_SUBSET
        )
        similarity_combination_element.similarity_measurement()
        df_similarity = similarity_combination_element.df_similarity
        df_dissimilarity = similarity_combination_element.df_dissimilarity
        df_uncertainty = similarity_combination_element.df_uncertainty
        classifier = InstanceBasedClassifier(
            df_data, df_similarity, df_dissimilarity, 
            df_uncertainty, n_gram_evidence=1
        )
        
        unobserved_combinations = generate_unobserved_combinations(
            df_data=df_data_train, elements=ELEMENTS, sizes=SIZES
        )
        
        y_pred, final_decisions = classifier.predict(
            np.array(unobserved_combinations), show_decision=True
        )
        
        df_result = report_recommendation_results(
            unobserved_combinations, final_decisions, df_data_test
        )
        df_result.to_csv("{}/high_recommend_model_{}.csv".format(OUTPUT_FOLDER, index_model))

    # Calculate the mean of ranking indices of HEAs in test set
    mean_ranking_HEA = calculate_mean_ranking_indices(df_data)
    
    # Visualize the ranking indices
    visualize_ranking_indices(mean_ranking_HEA)

OUTPUT_FOLDER = "../results/demo_3"
INPUT_DATA = "../data/HEA_data.ASMI16.csv"
N_SPLITS = 9
N_REPEATS = 3
N_CANDIDATE = 311 # Number of unobserved combinations in each cv
SIMILARITY_VALUE = 0.2 # parameter alpha
MAX_SIZE_SUBSET = 1
SIZES = [2] # size of unobserved combination
ELEMENTS = [
    'Mn', 'Fe', 'Nb', 'Tc', 'Au', 'Ta', 'W', 'Zr', 'Ni',
    'Mo', 'Rh', 'Co', 'Ir', 'Re', 'Pd', 'Ru', 'Cu', 'Pt',
    'Ti', 'Os', 'Cr', 'Hf', 'V', 'Ag', 'Al', "Si", "As"
]

if __name__ == '__main__':
    main()
    