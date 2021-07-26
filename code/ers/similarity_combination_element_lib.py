import pandas as pd 
import numpy as np
import math
try:
	from ers.MassFunction import MassFunction
except:
	from MassFunction import MassFunction
try:
	from scipy.misc import comb
except:
	from scipy.special import comb
import itertools
import datetime

def get_host(df_data, subset_1, subset_2):
	"""
	Returns hosts, which are intersections between pairwise instances containing subset_1 and subset_2.
	"""
	tmp_hosts = []
	query_string = " == 1 & ".join(subset_1) + " == 1" + "& length > {}".format(len(subset_1))
	df_tmp = df_data.query(query_string)
	for idx, rows in df_tmp.iterrows():
		host = set(rows['set_name'].split("|")).difference(subset_1)
		tmp_hosts.append(host)
	# Get evidence of similarity of subset2
	hosts = [set()]
	if len(subset_2) != 0:
		query_string = " == 1 & ".join(subset_2) + " == 1" + "& length > {}".format(len(subset_2))
		df_tmp = df_data.query(query_string)
		for idx, rows in df_tmp.iterrows():
			host = set(rows['set_name'].split("|")).difference(subset_2)
			if host in tmp_hosts:
				hosts.append(host)
	else:
		for host in tmp_hosts:
			query_string = " == 1 & ".join(host) + " == 1" + "& length == {}".format(len(host))
			df_tmp = df_data.query(query_string)
			if len(df_tmp.index) == 1:
				hosts.append(host)
	
	return hosts

def collect_similarity_evidence(df_data, sub_sets, similarity_value=0.2):
	"""
	Returns combined mass function of pieces of evidence about the similarity between two sub_set (elements or combinations of element).

	df_data: A data frame contains observed instances (Details of the data frame can be found in this link: https://doi.org/10.5281/zenodo.4557463).
	sub_sets: Two sub_set are considered to measure their similarity.
	similarity_value (or alpha): a hyperparameter is used to handle the conflict between pieces of evidence.
	"""
	final_decision = MassFunction(coreset={"similar", "dissimilar"})
	# Find common hosts
	subset_1, subset_2 = sub_sets
	if len(subset_1) < len(subset_2):
		subset_1, subset_2 = subset_2, subset_1
	if len(set(subset_1).intersection(set(subset_2))) != 0:
		return final_decision, subset_1, subset_2 
	hosts = get_host(df_data, subset_1, subset_2)
	
	for host in hosts:
		if len(host) == 0 and (len(subset_1)==0 or len(subset_2)==0):
			continue

		sub_material_1 = set(host).union(subset_1)
		query_string_1 = " == 1 & ".join(sub_material_1) + " == 1 " + "& length == {}".format(len(sub_material_1))
		df_sub_material_1 = df_data.query(query_string_1)
		label_sub_material_1 = df_sub_material_1["Label"].values

		sub_material_2 = set(host).union(subset_2)
		query_string_2 = " == 1 & ".join(sub_material_2) + " == 1 " + "& length == {}".format(len(sub_material_2))
		df_sub_material_2 = df_data.query(query_string_2)
		label_sub_material_2 = df_sub_material_2["Label"].values

		if len(df_sub_material_1.index) == 1 and len(df_sub_material_2.index) == 1:
			if label_sub_material_1 == label_sub_material_2:
				mass_function = MassFunction(
					source=[
						({"similar"}, np.float128(similarity_value))
					], 
					coreset={"similar", "dissimilar"}
				)
			else:
				mass_function = MassFunction(
					source=[
						({"dissimilar"}, np.float128(similarity_value))
					], 
					coreset={"similar", "dissimilar"}
				)
			final_decision = final_decision.combine(mass_function)
	return final_decision, subset_1, subset_2

def is_prediction_evidence(host, substitute, new_combination, 
	unk_score, similar_score, threshold
):
	"""
	Returns a boolean indicating that evidence satisfies conditions to be used for estimating.
	
	Conditions:
	1) Intersection between new_combination and substitute is empty set.
	2) Intersection between new_combination and host is empty set.
	3) The evidence has information (unk_score != 1).
	4) The new_combination not totally dissimilar to substitute (similar_score != 0).
	5) The new_combination is not empty set.
	"""
	return (len(new_combination.intersection(substitute)) == 0 and \
		len(new_combination.intersection(host)) == 0 and \
		unk_score != 1 and \
		similar_score != 0) and \
		len(new_combination) > 0
	
	# return (len(new_combination.intersection(host)) == 0 and \
	# 	unk_score != 1 and \
	# 	new_combination != substitute and \
	# 	similar_score != 0) and \
	# 	len(new_combination) > 0 and \
	# 	similar_score >= threshold

def collect_prediction_evidence( material, data, trace=False, 
	seperate_symbol="|", n_gram_evidence=1, threshold=0
):
	"""
	Returns the predicted labels of a new instance (material) and combined mass function of collected pieces of evidence.
	
	trace: If True, the function adds a list of pieces of evidence, which are used to estimate property of the new instance, to the output.
	seperate_symbol: symbol is used to separate elements in the instances of the data set.
	n_gram_evidence: max length of the combinations of element which are used to generate the new instance from observed instances based on substitution method.
	threshold: threshold is used as a filter to reduce pieces of evidence that are less important.
	"""
	df_data, df_similarity, df_dissimilarity, df_uncertainty = data
	atoms = material.split(seperate_symbol)
	final_decision = MassFunction(coreset={"High", "Low"})
	trace_list = []
	
	for size in range(n_gram_evidence):
		for substitute in itertools.combinations(atoms, size+1):
			substitute = set(substitute)
			host = set(atoms).difference(substitute)
			idx_atom = seperate_symbol.join(sorted(substitute))
			if idx_atom in df_similarity.index:
				for ith, similar_score, dissimilar_score, unk_score in zip(df_similarity.index.values, df_similarity[idx_atom].values, df_dissimilarity[idx_atom].values, 
						df_uncertainty[idx_atom].values
					):
					new_combination = set(ith.split(seperate_symbol)) if ith != '' else set()

					# Filter the pieces of evidence
					if is_prediction_evidence(host, substitute, new_combination,
						unk_score, similar_score, threshold
					):
						substituted_material = host.union(new_combination)
						query_string = " == 1 & ".join(substituted_material) + " == 1 " + " & length == {}".format(len(substituted_material))
						df_tmp = df_data.query(query_string)
						if len(df_tmp.index.values) == 1:
							mass_function = MassFunction(
								source=[
									({df_tmp["Label"].values[0]}, np.float128(similar_score*0.1)),
									({"High", "Low"}, np.float128(unk_score))
								], 
								coreset={"High", "Low"}
							)
							trace_list.append({
								"evidence": substituted_material,
								"host": host,
								"mass_function": mass_function
							})
							final_decision = final_decision.combine(mass_function)
	if trace:
		return material, final_decision, trace_list
	else:
		return material, final_decision

class SimilarityCombinationElement(object):
	"""
	A module, which is based on Dempster-Shafer theory, measures similarity between (1) elements, (2) element and combination of element, and (3) combinations of element.
	The module uses mass functions to model pieces of evidence about the similarity, which are collected from the data set.
	It should be noted that the instance in work is a multi-principal-element alloy (list of element).
	This module provides two running modes: single or parallel (using Spark).
	"""

	def __init__(
		self, df_data, partitions=None, 
		rage_size_subset=1, seperate_symbol="|", similarity_value=0.1,
		threshold=0.0
	):
		"""
		Creates a new object to measure the similarity.
		
		df_data: A data frame contains observed instances (Details of the data frame can be found in this link: https://doi.org/10.5281/zenodo.4557463).
		partitions: number of partitions related to Spark system (More details can be found in the documentation of Spark system).
		rage_size_subset: max length of the combinations of element which are considered about their similarity.
		seperate_symbol: symbol is used to separate elements in the instances of the data set.
		similarity_value (or alpha): a hyperparameter is used to handle the conflict between pieces of evidence.
		threshold: the threshold is used as a filter to reduce pieces of evidence that are less important.
		"""
		self.df_data = df_data
		self.seperate_symbol = seperate_symbol
		self.rage_size_subset = rage_size_subset
		self.partitions = partitions
		self.similarity_value = similarity_value
		self.is_measured = False
		self.threshold = threshold
		if not "set_name" in df_data.columns.values:
			raise ValueError()
		else:
			self.core_set = self.__find_coreset__()
			self.subsets = self.__generate_subsets__()

	def __find_coreset__(self):
		"""
		Return list of elements existing in instances of the data set.
		"""
		core_set = set()
		for label in self.df_data["set_name"].values:
			if len(core_set) == 0:
				core_set = set(label.split(self.seperate_symbol))
			else:
				core_set = core_set.union(set(label.split(self.seperate_symbol)))
		return list(core_set)

	def __generate_subsets__(self):
		"""
		Return list of subsets (elements or combinations of elements) is considered about their similarity with each other.
		"""
		subsets = []
		for size_subset in range(self.rage_size_subset + 1):
			for subset in itertools.combinations(self.core_set, size_subset):
				subsets.append(subset)
		return subsets

	def get_pairwise_subsets(self):
		"""
		Return list of pairwise considered subsets.
		"""
		pairwise_subsets = itertools.combinations(self.subsets, 2)
		return pairwise_subsets

	def __spark_similarity_measurement__(self):
		"""
		Return list of similarity between considered subsets.
		It should be noted that the function is implemented using the Spark system to accelerate runtime.
		"""
		pairwise_subsets = self.get_pairwise_subsets()
		from pyspark import SparkContext, SparkConf
		conf = (SparkConf().set("spark.driver.maxResultSize", "4g"))
		sc = SparkContext(
			appName="SimilarityCombinationElement", conf=conf
		)
		try:
			data = sc.broadcast(self.df_data)
			if comb(len(self.subsets),2) < self.partitions:
				rdd = sc.parallelize(pairwise_subsets, numSlices=comb(len(self.subsets),2))
			else:
				rdd = sc.parallelize(pairwise_subsets, numSlices=self.partitions)

			rs_hea = rdd.map(
				lambda sub_sets: collect_similarity_evidence(
					df_data=data.value, sub_sets=sub_sets, similarity_value=self.similarity_value
				)
			)

			results = rs_hea.collect()

		except KeyboardInterrupt:
			print("\nProgram terminated by Ctrl +C ")
		finally:
			sc.stop()
		
		return results

	def __similarity_measurement__(self):
		"""
		Return list of similarity between considered subsets.
		"""
		pairwise_subsets = self.get_pairwise_subsets()
		results = []
		for subsets in pairwise_subsets:
			final_decision, subset_1, subset_2 = collect_similarity_evidence(
				df_data=self.df_data, sub_sets=subsets, similarity_value=self.similarity_value
			)
			print("Evaluate {} and {}: {:.2f} similarity, {:.2f} dissimilarity, {:.2f} unknown".format(
				subset_1, subset_2, final_decision[frozenset({"similar"})],
				final_decision[frozenset({"dissimilar"})], final_decision[frozenset({"similar", "dissimilar"})])
			)
			results.append([final_decision, subset_1, subset_2])
		return results

	def __parse_results_to_df__(self, results):
		"""
		Return data frames containing data frames contain similarity, dissimilarity and uncertainty between considered subsets.
		"""
		n_subsets = len(self.subsets)
		columns_name = [self.seperate_symbol.join(sorted(k)) for k in self.subsets]
		
		similarity_matrix = np.zeros((n_subsets,n_subsets))
		np.fill_diagonal(similarity_matrix, 1)
		df_similarity = pd.DataFrame(similarity_matrix, columns=columns_name, index = columns_name)

		dissimilarity_matrix = np.zeros((n_subsets,n_subsets))
		df_dissimilarity = pd.DataFrame(dissimilarity_matrix, columns=columns_name, index = columns_name)
		
		unknown_matrix = np.ones((n_subsets,n_subsets))
		np.fill_diagonal(unknown_matrix, 0)
		df_uncertainty = pd.DataFrame(unknown_matrix, columns=columns_name, index = columns_name)

		for final_decision, subset_1, subset_2 in results:
			idx_1 = self.seperate_symbol.join(sorted(subset_1))
			idx_2 = self.seperate_symbol.join(sorted(subset_2))
			df_similarity[idx_1][idx_2] = final_decision[frozenset({"similar"})]
			df_similarity[idx_2][idx_1] = final_decision[frozenset({"similar"})]
			df_dissimilarity[idx_1][idx_2] = final_decision[frozenset({"dissimilar"})]
			df_dissimilarity[idx_2][idx_1] = final_decision[frozenset({"dissimilar"})]
			df_uncertainty[idx_1][idx_2] = final_decision[frozenset({"similar", "dissimilar"})]
			df_uncertainty[idx_2][idx_1] = final_decision[frozenset({"similar", "dissimilar"})]
		
		self.df_similarity = df_similarity
		self.df_dissimilarity = df_dissimilarity
		self.df_uncertainty = df_uncertainty

	def similarity_measurement(self, spark=False):
		"""
		Return list of similarity between considered subsets.

		spark: If True, the function is deployed on the Spark system to accelerate runtime.
		"""
		if spark:
			results = self.__spark_similarity_measurement__()
		else:
			results = self.__similarity_measurement__()
		self.__parse_results_to_df__(results)
		self.is_measured = True

	def to_csv(self, output_dir):
		"""
		Save calculated data frames to files.

		output_dir: a path of the destination folder to save the data frames.
		"""
		if self.is_measured:
			self.df_similarity.to_csv("{}/similarity.csv".format(output_dir))
			self.df_dissimilarity.to_csv("{}/dissimilarity.csv".format(output_dir))
			self.df_uncertainty.to_csv("{}/uncertainty.csv".format(output_dir))

class InstanceBasedClassifier(object):
	"""
	An instance-based classification module using Dempster-Shafer theory.
	The module uses mass function to model pieces of evidence about the property of the new instance, which are collected from the data set.
	It should be noted that the instance in the work is a multi-principal-element alloy (list of element).
	This module provides two running modes: single or parallel (using Spark).
	"""
	
	def __init__(self, df_data, df_similarity, df_dissimilarity, df_uncertainty, seperate_symbol="|", n_gram_evidence=2, partitions=256,
		threshold=0):
		"""
		Creates a new classifier.
		
		It is an object to classify new instance based on the observed data set.
		df_data: A data frame contains observed instances (Details of the data frame can be found in this link: https://doi.org/10.5281/zenodo.4557463).
		df_similarity, df_dissimilarity, df_uncertainty: data frames contain similarity, dissimilarity and uncertainty between (1) elements, (2) element and combination of element, and (3) combinations of element.
		seperate_symbol: symbol is used to separate elements in the instances of the data set.
		n_gram_evidence: max length of the combinations of element which are used to generate the new instance from observed instances based on substitution method.
		partitions: number of partitions related to Spark system (More details can be found in the documentation of Spark system).
		threshold: the threshold is used as a filter to reduce pieces of evidence that are less important.
		"""
		self.df_data = df_data
		self.df_similarity = df_similarity
		self.df_dissimilarity = df_dissimilarity
		self.df_uncertainty = df_uncertainty
		self.seperate_symbol = seperate_symbol
		self.n_gram_evidence = n_gram_evidence
		self.partitions = partitions
		self.threshold = threshold

	def predict(self, materials, trace=False, spark=False, show_decision=False):
		"""
		Returns the predicted labels of new instances (materials).
		
		trace: If True, the function adds lists of pieces of evidence, which are used to estimate properties of the new instances, to the output.
		show_decision: If True, the function adds the combined mass function of collected pieces of evidence to the output.
		spark: If True, the function is deployed on the Spark system to accelerate runtime.
		"""
		y_pred = []
		trace_pred = []
		final_decisions = []
		if spark:
			from pyspark import SparkContext, SparkConf
			conf = (SparkConf().set("spark.driver.maxResultSize", "4g"))
			sc = SparkContext(
				appName="Prediction", conf=conf
			)
			try:
				data = sc.broadcast((self.df_data, self.df_similarity, self.df_dissimilarity, self.df_uncertainty))
				if len(materials) < self.partitions:
					rdd = sc.parallelize(materials, numSlices=len(materials))
				else:
					rdd = sc.parallelize(materials, numSlices=self.partitions)

				rs_hea = rdd.map(
					lambda material: collect_prediction_evidence(
						data=data.value, material=material, n_gram_evidence=self.n_gram_evidence,
						threshold=self.threshold
					)
				)
				y_pred = ["" for k in materials]
				results = rs_hea.collect()
				for material, final_decision in results:
					if final_decision[frozenset({"High"})] > final_decision[frozenset({"Low"})]:
						y_pred[int(np.where(materials==material)[0])] = "High"
					else:
						y_pred[int(np.where(materials==material)[0])] = "Low"
					final_decisions.append(final_decision)

			except KeyboardInterrupt:
				print("\nProgram terminated by Ctrl +C ")
			finally:
				sc.stop()
			if show_decision:
				return y_pred, final_decisions
			else:
				return y_pred
		else:
			data = (self.df_data, self.df_similarity, self.df_dissimilarity, self.df_uncertainty)
			for material in materials:
				if trace:
					material, final_decision, trace_list = collect_prediction_evidence(
						data=data, material=material, n_gram_evidence=self.n_gram_evidence,
						trace=trace, threshold=self.threshold
					)
					trace_pred.append(trace_list)
				else:
					material, final_decision = collect_prediction_evidence(
						data=data, material=material, n_gram_evidence=self.n_gram_evidence,
						threshold=self.threshold
					)
				if final_decision[frozenset({"High"})] > final_decision[frozenset({"Low"})]:
					y_pred.append("High")
				else:
					y_pred.append("Low")
				final_decisions.append(final_decision)
			if show_decision and trace:
				return y_pred, final_decisions, trace_pred
			elif trace:
				return y_pred, trace_pred
			elif show_decision:
				return y_pred, final_decisions
			else:
				return y_pred