import numpy as np 
import itertools as itt 
from scipy.spatial import KDTree
import time

from utils.wbFingerprintUtils import print_time
import params.config as config




"""
Class comparing new Histogram of Oriented Gradient (HOG) signatures against a database of signatures. Uses L2-distance KDTree.
"""

def multiplier(x, score_type, alpha=2, max_matches=20, accum_type="weighted"):
	n = len(x)
	max_matches = np.minimum(n, max_matches)
	sorted_scores = np.sort(x, axis=0)

	if score_type == "ratios":
		sorted_scores = sorted_scores[::-1]
		sorted_scores = sorted_scores[:max_matches]
	if n == 1:
		stop = n
	else:
		stop = max_matches
		#diff = sorted_scores[:max_matches-1] / sorted_scores[1:max_matches]
		#stop = np.argmax(diff)+1
	if accum_type == "weighted":
		weight = (alpha * np.arange(stop)) + 1 
		score = np.sum((sorted_scores[:stop].reshape(weight.shape) / weight))
	elif accum_type == "mean":
		score = np.mean(sorted_scores[:stop])
	else:
		raise NameError(accum_type +" is not a recognized score accumulation type")

	return score




class wbForensicsHOG(object):
	"""
	Class used to check new western-blot fingerprints against a database of fingerprints. Compares Histogram of 
	Oriented Gradient signatures using L2-distance in a KDTree. Copies are detected by a threshold specified by 
	the user.
	"""

	def __init__(self, Database, d_T=1, r_T=1, score_T=1):
		self.Database = Database # Database of signatures to compare the new fingerprint against
		#thresholds
		self.d_T = d_T 
		self.r_T = r_T
		self.score_T = score_T
		# initialized indices
		self.b_idx = 0
		self.batch_tracker_idx = 0
		# KD-tree for whole database
		self.Full_Tree = KDTree(self.Database["feature_vectors"], len(self.Database["feature_vectors"])+1)


	def next_figure(self):
		"""
		sweeps over figures and returns a batch of features from the next figure in the database
		"""
		figure_indices =  np.unique(self.Database["figure_number"])
		stop_idx = len(figure_indices) - 1
		if self.batch_tracker_idx > stop_idx:
			return(None, None)

		batch_indices = np.where(np.array(self.Database["figure_number"]) == figure_indices[self.batch_tracker_idx])
		batch = self.Database["feature_vectors"][batch_indices]
		self.batch_tracker_idx += 1

		return(batch, batch_indices[0])


	def KDTree_pairs(self, leaf_size=10):
		"""
		finds all pairs within a distance self.L2_T of a batch of signatures

		Parameters:f
		------------
		leaf_size: int
			The number of points at which the algorithm switches over to brute-force. Has to be positive
		Returns:
		------------ 
		N/A
		"""

		# initialize the pair array
		self.pairs = []; self.ratios = []; self.dists = []
		# calculate the number of epochs from batch method
		epochs = len(np.unique(self.Database["figure_number"]))

		time.clock()
		self.pair_counter = 0
		for i in range(epochs):

			# keep track of pairs recorded and time
			# if (i) % 10 == 0:
			# 	print(print_time(time.time()), "|| %s of %s batches completed; %s pairs extracted"
			# 		%(i, epochs, self.pair_counter))

			# get new batch and respected indices
			batch, batch_indices = self.next_figure()
			if batch is None or len(batch_indices)<2:
				self.pairs.append([]); self.ratios.append([]); self.dists.append([])
				continue

			# build the kd-tree from the the batch
			Tree = KDTree(batch, leaf_size)

			if config.query_method == "ALL_PAIRS":
				all_pairs = np.array(list(itt.combinations(range(len(batch_indices)), 2)))
				all_features = batch[all_pairs]
				dists = np.linalg.norm(np.diff(all_features, axis=1), axis=2)

				ratios = find_next_dist(Tree, batch, all_pairs, dists, k=len(batch)+1)
				pairs = batch_indices[all_pairs]

				if config.r_T is not None:
					thresh_indices = np.where(ratios > config.r_T)[0]
					ratios = ratios[thresh_indices]
					dists = dists[thresh_indices]
					pairs = pairs[thresh_indices]

				self.pairs.append(pairs); self.dists.append(dists); self.ratios.append(ratios)
				
			elif config.query_method == "NN":
				# find k nearest neighbors for each signature with the figure
				d, tree_i = Tree.query(batch, k=2)

				# initialize final distance array
				d_final = d[:,1]

				d_ratio = np.zeros(d.shape[0])
				# find nearest neighbors in whole 
				d2, full_tree_i = self.Full_Tree.query(batch, k=3)


				# # decide if the nearest neighbor is within or outside the same figure
				# within_fig_truth = np.where(full_tree_i[:,1] - batch_indices[tree_i[:,1]] == 0) # indices if nearest neighbor within same figure
				# outside_fig_truth = np.where(full_tree_i[:,1] - batch_indices[tree_i[:,1]] != 0) # indices if nearest neighbor not within same figure
				# # compute ratios
				# d_ratio[within_fig_truth] = d2[:,2][within_fig_truth] / d[:,1][within_fig_truth] # take ratio if within same figure
				# d_ratio[outside_fig_truth] = 1 #otherwise set to 1
				# # make sure there are no NaNs
				# d_ratio[within_fig_truth][np.where(d2[:,2][within_fig_truth]==0)] = 0
				# d_ratio[np.where(d[:,1]==0)] = 1


				d_ratio = d2[:,2]/d[:,1]
				d_ratio[np.where(d[:,1]==0)] = 1


				# find unique pairs from the k-nearest neighbors
				batch_pairs = np.hstack((batch_indices.reshape(len(batch_indices),1),batch_indices[tree_i[:,1, None]])) 
				#batch_pairs = batch_indices[np.array(list(Tree.query_pairs(self.match_T)), dtype=int)]

				batch_pairs = np.sort(batch_pairs, axis=1) # make sure smaller index is first
				batch_pairs, unique_indices = np.unique(batch_pairs, axis=0, return_index=True) # only find unique pairs

				d_ratio = d_ratio[unique_indices] # distance ratios of unique pairs
				d_final = d_final[unique_indices]

				# threshold the match distances by dist and ratio thresholds
				# thresh_indices = np.where((d_final <= self.d_T)*(d_ratio >= self.r_T))
				# batch_pairs = batch_pairs[thresh_indices]
				# d_ratio = d_ratio[thresh_indices]
				# d_final = d_final[thresh_indices]

				#append the pair indices and distance ratios
				self.pairs.append(batch_pairs); self.ratios.append(d_ratio); self.dists.append(d_final)
			else:
				raise NameError(QUERRY_METHOD + " is not a recognized query method.")

			# keep track of the amount of pairs recorder
			#self.pair_counter += len(batch_pairs)


	def d_rank(self, pairs, ratios, distances):
		"""
		ranks unique image pairs based on distance ratio scores between matched features of the two images

		Parameters:
		--------------
		pairs: python list of lists
			list of batched pairs per figure (figures[pair_indices])
		distances: python list of lists
			list of distance scores per figure (figures[dist_scores])

		"""

		# initialize the output directory
		self.Dist_Rank = {"figure_number": [], # indices of figures in the database
					  	  "image_pairs": [], # indices of image pairs
					  	  "image_match_ratios": [], # distance ratio scores
					  	  "image_match_dists": [],
					  	  "image_match_pairs": [], # indices of paired features per image pair
					  	  "image_feature_scores": []
					  	  }


		unique_fig_nums = np.unique(self.Database["figure_number"]) # all figure numbers
		for idx, fig_num in enumerate(unique_fig_nums):

			# extract the indices and dist ratios for the pairs in the figure   
			feature_pairs = pairs[idx] # pair indices 
			feature_ratios = ratios[idx] # pair distance ratio scores # distances[idx]
			feature_distances = distances[idx] # np.max(distances[idx]) - distances[idx]	
			assert len(feature_distances) == len(feature_ratios)
	
			if feature_pairs is None or len(feature_pairs) == 0:
				continue

			# find indices of paired and single images after nearest-neighbor filtering
			image_pairs = np.array(self.Database["image_idx"])[feature_pairs] # find image indices associated with each pair
			image_queries = np.unique(image_pairs.flatten()) # unique image indices after nearest-neighbor filterering

			if image_queries is None or len(image_queries) == 0:
				continue

			# record the minumum pair distances for each image pair 
			image_match_dists = []; image_match_pairs = []; image_match_ratios=[];
			image_feature_scores=[]
			for image_pair in np.unique(image_pairs, axis=0):
				# find specific image pair feature indices
				image_pair_indices = np.where(np.all(np.abs(image_pairs - image_pair) == 0, axis=1))

				# find corresponding image pair ratios and distance
				image_feature_distances = feature_distances[image_pair_indices]
				image_feature_ratios = feature_ratios[image_pair_indices]
				
				# compute match score
				image_match_ratio = multiplier(image_feature_ratios, "ratios", 
											   alpha=config.multiplier_num_sum, accum_type=config.accum_type)
				image_match_dist = multiplier(image_feature_distances, "dists", 
											  alpha=config.multiplier_num_sum, accum_type=config.accum_type)
				image_match_pair = feature_pairs[image_pair_indices]

				# filter by ratio
				# if image_match_ratio < self.score_T:
				# 	continue

				image_match_pairs.append(image_match_pair); image_match_dists.append(image_match_dist); image_match_ratios.append(image_match_ratio); 
				image_feature_scores.append(image_feature_ratios)
				#image_feature_scores.append(image_feature_distances)
			
			# find distance ratio rankings (d-rank)

			# append output dictionary info
			if len(image_match_pairs) > 0:
				self.Dist_Rank["figure_number"].append(fig_num) # index of figure in database
				self.Dist_Rank["image_pairs"].append(np.unique(image_pairs, axis=0)) # indices of the image pairs
				self.Dist_Rank["image_match_ratios"].append(image_match_ratios) # distance ratio raw score
				self.Dist_Rank["image_match_dists"].append(image_match_dists)
				self.Dist_Rank["image_match_pairs"].append(image_match_pairs) #
				self.Dist_Rank["image_feature_scores"].append(image_feature_scores)


	def assign_match_label(self, pairs, ratios, distances, ASSIGNMENT):
		"""
		ranks unique blot pairs based on distance ratio scores between matched features of the two images

		Parameters:
		--------------
		pairs: python list of lists
			list of batched pairs per figure (figures[pair_indices])
		distances: python list of lists
			list of distance scores per figure (figures[dist_scores])

		"""

		# initialize the output directory
		self.truth_values = {"figure_number": [], # indices of figures in the database
				 			 "pairs": [], # pair indices
				 			 "ratios": [], # distance ratio scores
				 			 "distances": [], # distance scores
				 			 "match_nums": [],
				 			 "y_truth": []} # reference set match labels. Given value of -1 if no match assign


		if ASSIGNMENT == "feature":
			unique_fig_nums = np.unique(self.Database["figure_number"]) # all figure numbers
		elif ASSIGNMENT == "image":
			unique_fig_nums = np.unique(self.Dist_Rank["figure_number"])
		else:
			raise NameError(ASSIGNMENT + " is not a recognized assignment type")

		for idx, fig_num in enumerate(unique_fig_nums):

			# extract the indices and dist ratios for the pairs in the figure   
			feature_pairs = pairs[idx] # pair indices 
			feature_ratios = ratios[idx]# (np.max(distances[idx]) + 1) - distances[idx] # pair distance ratio scores # distances[idx]
			feature_distances = distances[idx]		
			#assert len(feature_distances) == len(feature_ratios)
	
			if feature_pairs is None or len(feature_pairs) == 0:
				continue

			# assign truth value to matches via reference set labels
			for i, pair in enumerate(feature_pairs):
				# assign truth values
				## TODO: problem with indexing. index expects feature index, but image index is given
				if ASSIGNMENT == "feature":
					idx1 = pair[0]
					idx2 = pair[1]

					# find intersection of sets
					pair_set1 = set(self.Database["reference_label"][idx1]) 
					pair_set2 = set(self.Database["reference_label"][idx2])
					intersection = pair_set1 & pair_set2

					if intersection and -1 not in intersection:
						self.truth_values["match_nums"].append(list(intersection))
						self.truth_values["y_truth"].append(1)
					else:
						self.truth_values["match_nums"].append([-1])
						self.truth_values["y_truth"].append(0)

				else:
					idx1 = pair[0][0]
					idx2 = pair[0][1]

					# find intersection of sets
					pair_set1 = set(self.Database["reference_label"][idx1]) 
					pair_set2 = set(self.Database["reference_label"][idx2])
					intersection = pair_set1 & pair_set2

					if self.Database["image_idx"][idx1] == self.Database["image_idx"][idx2]:
						if len(intersection) > 1 and "self_overlap" in intersection and -1 not in intersection:
							self.truth_values["match_nums"].append(list(intersection))
							self.truth_values["y_truth"].append(1)
						else:
							self.truth_values["match_nums"].append([-1])
							self.truth_values["y_truth"].append(0)

					else:
						if intersection and -1 not in intersection and "self_overlap" not in intersection:
							self.truth_values["match_nums"].append(list(intersection))
							self.truth_values["y_truth"].append(1)
						else:
							self.truth_values["match_nums"].append([-1])
							self.truth_values["y_truth"].append(0)
					

				self.truth_values["figure_number"].append(fig_num)
				self.truth_values["pairs"].append(pair)
				self.truth_values["ratios"].append(feature_ratios[i])
				self.truth_values["distances"].append(feature_distances[i])


	
			
def find_next_dist(Tree, batch, all_pairs, dists, k):
	d, i = Tree.query(batch, k=k)

	ratios = []
	for p, dist in zip(all_pairs, dists):
		dr1 = d[p[0]][np.where(d[p[0]] > dist)][0]
		dr2 = d[p[1]][np.where(d[p[1]] > dist)][0]
		dr = np.mean([dr1, dr2])

		if np.isinf(dr):
			ratios.append(1)
		else:
			ratios.append(float(dr/(dist +  np.finfo(float).eps)))


	return( np.array(ratios).reshape(dists.shape) )













