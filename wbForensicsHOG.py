import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import math
from scipy.spatial import KDTree
import time
import wbFingerprint as wb


"""
Class comparing new Histogram of Oriented Gradient (HOG) signatures against a database of signatures. Uses L2-distance KDTree.
"""
# DATABASE_INPUT_FILE = "testFigures/nature_wb_output/fingerprintDatabase_testNewSegMeth.npy"
# PAIRS_OUTPUT_FILE = "testFigures/nature_wb_output/matches_final.npy"
# VOTES_OUTPUT_FILE = "testFigures/nature_wb_output/votes_final.npy"

def load_database(path_to_file):
	Database = np.load(path_to_file).item()
	return(Database)

class wbForensicsHOG(object):
	"""
	Class used to check new western-blot fingerprints against a database of fingerprints. Compares Histogram of Oriented Gradient 
	signatures using L2-distance and KDTree. Copies are detected by a threshold specified by the user.
	"""

	def __init__(self, signature, Database, L2_T):
		self.signature = signature # new fingerprint (contains HOG signature and other location information)
		self.Database = Database # Database of signatures to compare the new fingerprint against
		self.L2_T = L2_T # L2-distance threshold. Distnaces above this value are classified as a copy.
		self.b_idx = 0
		self.paper_idx = 0

		# organize by paper
		self.papers_all = []
		for tag in self.Database["figure_tags"]:
			self.papers_all.append(tag["article_title"])
		self.papers_unique = np.unique(self.papers_all)

	def next_paper(self):
		stop_idx = len(self.papers_unique) - 1
		if self.paper_idx > stop_idx:
			return(None, None)

		batch_indices = np.where(np.array(self.papers_all)==self.papers_unique[self.paper_idx])
		batch = self.Database["feature_vectors"][batch_indices]
		self.paper_idx += 1

		return(batch, batch_indices[0])


	def next_batch(self, batch_size, overlap):
		stop_idx = len(self.Database["feature_vectors"])
		if self.b_idx > stop_idx:
			return(None, None)

		batch = self.Database["feature_vectors"][self.b_idx: np.minimum(self.b_idx + batch_size, stop_idx)]
		self.b_idx = self.b_idx + (batch_size - overlap)

		old_idx = self.b_idx - (batch_size - overlap)
		return(batch, old_idx)


	def KDTree_pairs(self, leaf_size, batch_size, overlap):
		"""
		creates the KDTree object

		Parameters:
		------------
		leaf_size: int
			The number of points at which the algorithm switches over to brute-force. Has to be positive
		k: int
			The number of nearest neighbors to return.
		Returns:
		------------ 
		"""
		self.pairs = []

		# epochs = math.ceil(len(self.Database["feature_vectors"]) / (batch_size - overlap))
		epochs = len(self.papers_unique)

		time.clock()
		for i in range(epochs):
			# batch, shift_index = self.next_batch(batch_size, overlap)
			if (i) % 100 == 0:
				print(round(time.clock()), "secs: %s of %s batches completed; %s pairs extracted"%(i, epochs, len(self.pairs)))

			batch, batch_indices = self.next_paper()
			if batch is None:
				break
			Tree = KDTree(batch, leaf_size)
			batch_pairs = batch_indices[np.array(list(Tree.query_pairs(self.L2_T)), dtype=int)]

			if len(self.pairs) == 0:
				self.pairs = batch_pairs 
			else:
				if len(batch_pairs) > 0:
					self.pairs = np.vstack((self.pairs, batch_pairs))
				else: 
					continue

	def pair_scoring(self, pairs, d):
		""" Scoring goals:
			- each best match pair will get assigned a score based on the following criteria
			
			Score:
			- best matches that are less similar (more unique) compared to other signatures get a higher score depending on level of uniqueness
				- level of uniqueness depends on percentage of other matches (not best-match) in database

			Multiplier:
			- sequences of best matches get higher multiplier per match in sequence

		"""
		self.Votes = {"fig_idx": [], "score": [], "multiplier":[], "pairs":[]} # intialize scoring table
		Tree = KDTree(Database["feature_vectors"]) # kd-tree of entire database 

		for match in pairs:
			# uniqueness score calculation
			nearest_neighbors_1 = len(Tree.query_ball_point(Database["feature_vectors"][match[0]], self.L2_T)) # number of neighbors within distance L2_T of match1 
			nearest_neighbors_2 = len(Tree.query_ball_point(Database["feature_vectors"][match[1]], self.L2_T)) # number of neighbors within distance L2_T of match2
			# TODO:
				# weight each nearest neighbor by how "close" they are in image, figure, paper, journal, etc. heirarchy 
			score = 1 / np.mean([nearest_neighbors_1, nearest_neighbors_2]) # score each match as 1 / average of (# nearest neighbors within L2_T)
			self.Votes["score"].append(score)

			# sequence multiplier
			match_locs = self.Database["image_locs"][match]
			match_centers = self.Database["centroids"][match]

			indices = batch_indices[0][np.where(np.sum(self.Database["image_locs"] - match_locs[0], axis=1) == 0)]
			blot_centers = self.Database["centroids"][indices]
			for i, b_center in enumerate(blot_centers):
				dist = np.linalg.norm(match_centers[0] - b_center)
				if dist > 0:
					


			batch_indices_2 = np.where(np.array(self.papers_all)==np.array(self.papers_all)[match[1]])
			batch_2 = self.Database["image_locs"][batch_indices_2]
			batch2 - match_locs[1]


			for i, imae_loc in enumerta(batch2):




			






			





		

		## sequence multiplier
		"""
		Second Check: there is more than one seperate match within the same image,
			then measure dist between centroids
			otherwise skip
		Third Check: pairs of respected match are also within same image,
			then measure dist between centroids 
			otherwise skip
		Fourth Check: dist measurements preserved,
			then uniqueness_score x (# matches in sequence)
		
		"""





		# calculate votes
		dist_vote = (np.abs(np.diff(pairs, axis=1)) < d).astype(int).reshape(shape)
		fig_vote = (np.diff(np.array(self.Database["figure_idx"])[pairs], axis=1) == 0).astype(int).reshape(shape)
		img_vote = (np.sum(np.sum(np.diff(np.array(self.Database["image_locs"])[pairs], axis=1), axis=1), axis=1) == 0).astype(int).reshape(shape)

		for fig_idx in np.unique(figure_indices):
			indices = np.where(figure_indices == fig_idx)
			mult = (np.sum(dist_vote[indices]), np.sum(fig_vote[indices]), np.sum(img_vote[indices]))
			vote = mult[0] + 1.5*mult[1] + 2*mult[2]
			pair = np.array(self.intersecting_pairs([list(p) for p in pairs[indices]]))

			self.Votes["fig_idx"].append(fig_idx)
			self.Votes["vote"].append(vote)
			self.Votes["mult"].append(mult)
			self.Votes["pairs"].append(pair)  

	def intersecting_pairs(self, pairs):
		"""
		merge intersecting sets
		"""
		sets = [set(lst) for lst in list(pairs) if lst]
		merged = 1
		while merged:
			merged = 0
			results = []
			while sets:
				common, rest = sets[0], sets[1:]
				sets = []
				for x in rest:
					if x.isdisjoint(common):
						sets.append(x)
					else:
						merged = 1
						common |= x
				results.append(common)
			sets = results
		return(sets)


def view(DATABASE_INPUT_FILE,VOTES_OUTPUT_FILE, PATH_TO_FILE, FOLDER_NAME):
	Database = load_database(DATABASE_INPUT_FILE)
	Votes = load_database(VOTES_OUTPUT_FILE)
	figure_data = np.load(PATH_TO_FILE)
	#print(figure_data)

	Votes["flag"] = []

	for idx, fig_idx in enumerate(Votes["fig_idx"]):
		figure = wb.WesternBlot(figure_data[fig_idx], label="images", folder=FOLDER_NAME).figure.copy()
		count_outside_figure = 0
		if Votes["vote"][idx] == 0:
			Votes["flag"].append(0)
			continue
		for match in Votes["pairs"][idx]:
			color = np.random.randint(0,255,size=3)
			for i in list(match):
				if Database["figure_idx"][i] != fig_idx:
					count_outside_figure += 1
					continue
				i_loc = Database["image_locs"][i]
				b_loc = Database["blot_locs"][i]

				image = figure[i_loc[1]:i_loc[1]+i_loc[3], i_loc[0]:i_loc[0]+i_loc[2]]
				cv2.rectangle(image,
							  pt1 = (b_loc[0], b_loc[1]),
							  pt2 = (b_loc[0]+b_loc[2], b_loc[1]+b_loc[3]),
							  color = (int(color[0]), int(color[1]), int(color[2])),
							  thickness = 1)

		fig = plt.figure(figsize=(20,20))
		plt.imshow(figure)
		plt.title("Figure #: %s; Votes: %s; Number of other matches: %s"%(fig_idx, Votes["vote"][idx], count_outside_figure))
		plt.draw()
		plt.pause(1)
		response = str(input("Figure "+str(fig_idx)+": Flag for forgery, input 1; otherwise, input 2: "))
		plt.close(fig)

		answered = 1
		while answered:
			if response == '1':
				Votes["flag"].append(1)
				answered = 0 
			elif response == '2':
				Votes["flag"].append(0)
				answered = 0
			else:
				response = str(input("Try again: Figure "+str(fig_idx)+": Flag for forgery, input 1; otherwise, input 2: "))

	np.save(VOTES_OUTPUT_FILE, Votes)


			



		


def main(DATABASE_INPUT_FILE, PAIRS_OUTPUT_FILE, VOTES_OUTPUT_FILE):
	# initialize
	Database = load_database(DATABASE_INPUT_FILE)
	WBF = wbForensicsHOG(None, Database, .10)

	# # find pairs using k-nearest neighbors
	WBF.KDTree_pairs(10,1000,100)
	# np.save(PAIRS_OUTPUT_FILE, WBF.pairs)
	# print("Pairs saved in drive")

	# # # count votes on pairs
	# WBF.pair_votes(WBF.pairs, 40)
	# np.save(VOTES_OUTPUT_FILE, WBF.Votes)
	# print("Votes saved in drive")


if __name__ == "__main__":
 	#main()
	#view()

	# BMC_output schedule

	# batch1 9/20/2017
		# L2 = 0.10
		# interval = 1000
		# overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch1.npy",
	#      "testFigures\BMC_output\BMC_output_pairs_batch1", 
	#      "testFigures\BMC_output\BMC_output_votes_batch1" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch1.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch1.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch1(0,999).npy",
	# 	 "BMC_output")

	# batch2 9/21/2017
		# L2 = 0.10
		# interval = 1000
		# overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch2.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch2", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch2" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch2.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch2.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch2(1000,1999).npy",
	# 	 "BMC_output")

	# batch3 9/20/2017
		# L2 = 0.10
		# interval = 1000
		# overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch3.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch3", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch3" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch3.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch3.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch3(2000,2999).npy",
	# 	 "BMC_output")

	# batch4 9/21/2017
		# L2 = 0.10
		# interval = 1000
		# overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch4.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch4", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch4" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch4.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch4.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch4(3000,3999).npy",
	# 	 "BMC_output")

	# batch5 9/21/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch5.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch5", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch5" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch5.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch5.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch5(4000,4999).npy",
	# 	 "BMC_output")

	# batch6 9/21/2017
	#	L2 = 0.10
	#	interval = 1000
	#	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch6.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch6", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch6" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch6.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch6.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch6(5000,5999).npy",
	# 	 "BMC_output")

	# batch7 9/22/17
	#	 L2 = 0.10
	#	 interval = 1000
	#	 overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch7.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch7", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch7" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch7.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch7.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch7(6000,6999).npy",
	# 	 "BMC_output")

	# batch8 9/22/17
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch8.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch8", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch8" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch8.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch8.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch8(7000,7999).npy",
	# 	 "BMC_output")

	# batch9 9/22/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch9.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch9", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch9" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch9.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch9.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch9(8000,8999).npy",
	# 	 "BMC_output")

	# batch10 9/22/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch10.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch10", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch10" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch10.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch10.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch10(9000,9999).npy",
	# 	 "BMC_output")

	# batch12 9/26/2017 error ... 
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch12.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch12", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch12" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch12.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch12.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch12(11000,11999).npy",
	# 	 "BMC_output")

	# batch13 9/26/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# # 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch13.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch13", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch13" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch13.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch13.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch13(12000,12999).npy",
	# 	 "BMC_output")

	# batch14 9/26/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch14.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch14", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch14" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch14.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch14.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch14(13000,13999).npy",
	# 	 "BMC_output")

	# batch15 9/26/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	# main("testFigures\BMC_output\BMC_output_fingerprint_batch15.npy",
	# 	 "testFigures\BMC_output\BMC_output_pairs_batch15", 
	# 	 "testFigures\BMC_output\BMC_output_votes_batch15" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch15.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch15.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch15(14000,14999).npy",
	# 	 "BMC_output")

	# batch16 9/26/2017
	# 	L2 = 0.10
	# 	interval = 1000
	# 	overlap = 100
	main("testFigures\BMC_output\BMC_output_fingerprint_batch16.npy",
		 "testFigures\BMC_output\BMC_output_pairs_batch16", 
		 "testFigures\BMC_output\BMC_output_votes_batch16" )
	# view("testFigures\BMC_output\BMC_output_fingerprint_batch16.npy",
	# 	 "testFigures\BMC_output\BMC_output_votes_batch16.npy",
	# 	 "testFigures/BMC_output/BMC_output_batch16(15000,15999).npy",
	# 	 "BMC_output")















