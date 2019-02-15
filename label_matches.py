import matplotlib.pyplot as plt 
import numpy as np 
import cv2

from wbFingerprint import WesternBlot
from utils.wbFingerprintUtils import get_blot_points




class AnnotateMatches(object):
	""" class for labeling match locations in a figure """

	def __init__(self, Figures, FOLDER_NAME, OUTPUT_FILE):
		"""
		INPUTS:
		----------
		Figures: python dictionary
			dictionary housing the figures and figure tags
		FOLDER_NAME: str
			name of the corresponding figure folder
		OUTPUT_FILE:
			name of the output file to store the match locations
		"""

		# initialize
		self.match_points = {"figure_number":[], "points":[], "match_num":[]}
		self.match_num = 0
		self.paper_tracker = []

		# loop over figures and record match locations
		for i, Figure in enumerate(Figures):
			WB = WesternBlot(dir_dictionary=Figure, label="images", folder=FOLDER_NAME) # create western blot object

			# only record matches for papers containing them
			if WB.tags["label"]:
				paper_idx = WB.tags["paper_idx"] 
				print("DOI: {}; Paper index: {}".format(WB.tags["DOI"], paper_idx))

				self.assign_match(WB.figure, i, paper_idx) 

				np.save(OUTPUT_FILE, self.match_points)

		print("All figures annotated")



	def assign_match(self, figure, fig_num, paper_num):
		"""
		Interactive tool to assign labels to matches (i.e. true copy or not)

		INPUTS:
		---------
		figure: numpy array
			figure to label 
		fig_num: int
			index of figure in database
		paper_num: int
			index of paper in database
		
		"""
		if paper_num in self.paper_tracker:
			return()

		figure_unlabeled = figure.copy()
		plt.figure()
		plt.imshow(figure_unlabeled)
		plt.show()
		continue_fig = int(input("To begin entering matches press 1. To skip press 0. To skip paper press 2: "))

		if continue_fig == 1:

			continue_next = 1
			while continue_next:
				points = self.get_match(figure_unlabeled)

				self.match_points["figure_number"].append(fig_num)
				self.match_points["points"].append(points)
				self.match_points["match_num"].append(self.match_num)
				self.match_num += 1

				continue_next = int(input("To continue assigning matches press 1. To go to next figure press 0: "))

		elif continue_fig == 2:
			self.paper_tracker.append(paper_num)
			return()

		else:
			return()


	def get_match(self, figure):
		"""
		Interactive tool for labeling of matched features. Prompts you to select two features that should be assigned a match label, then returns an updated image
		"""
		# get match points
		print("Click on top left corner of two features that match")
		plt.imshow(figure)
		p1, p2 = plt.ginput(2, timeout=300)
		plt.close()

		# re-label current figure
		cv2.rectangle(figure, (int(p1[0]), int(p1[1])), (int(p1[0]+15), int(p1[1]+15)), color=(255,0,0), thickness=-1)
		cv2.rectangle(figure, (int(p2[0]), int(p2[1])), (int(p2[0]+15), int(p2[1]+15)), color=(255,0,0), thickness=-1)

		return(np.array(p1+p2))


class AssignMatches(object):
	"""
	Assigns segmented bounding boxes to match labels.
	"""

	def __init__(self, database, reference_labels, figures, FOLDER_NAME):
		"""
		INPUTS:
		----------
		database: python dictionary
			database containing fingerprint information
		reference_labels: python dictionary
			contains hand-anooted match labels
		figures: python dictionary
			contains figure information
		FOLDER_NAME: str
			name of folder containing raw figures
		"""
		self.Database = database
		self.Figures = figures
		self.Reference_Labels = reference_labels
		self.FOLDER_NAME = FOLDER_NAME

		self.Database["reference_label"] = [[-1]] * len(self.Database["figure_number"])

	def assign(self, ASSIGNMENT):
		"""
		Assigns reference labels to segmented bounding boxes. Updates database with new category: "reference_label"
		
		INPUTS:
		---------
		ASSIGNMENT: str
			assignment type.
				"image" - assign reference labels for image locs
				"feature" - assign reference labels for feature locs
		"""
		for i, fig_idx in enumerate(np.unique(self.Reference_Labels["figure_number"])):
			figure = WesternBlot(dir_dictionary = self.Figures[fig_idx], label="images", folder=self.FOLDER_NAME).figure.copy()

			# reference labels for given batch
			reference_indices = np.where(np.array(self.Reference_Labels["figure_number"]) == fig_idx)
			reference_labels_fig = {}
			for key in self.Reference_Labels.keys():
				reference_labels_fig[key] = np.array(self.Reference_Labels[key])[reference_indices]

			# image locs and blot locs for given batch
			feature_indices = np.where(np.array(self.Database["figure_number"]) == fig_idx)[0]

			for feature_i in feature_indices:
				if ASSIGNMENT == "feature":
					i_loc = self.Database["image_locs"][feature_i]
					b_loc = self.Database["blot_locs"][feature_i]
					rec = get_blot_points(i_loc, b_loc)

				elif ASSIGNMENT == "image":
					rec = self.Database["image_locs"][feature_i]

				else:
					raise NameError(ASSIGNMENT + " is not a recognized assignment type")

				match_labels = []
				for pnt, match_num in zip(reference_labels_fig["points"], reference_labels_fig["match_num"]):
					if ASSIGNMENT == "feature":
						if self.overlap_truth(pnt[:2], rec, figure.shape) or self.overlap_truth(pnt[2:], rec, figure.shape):
							match_labels.append(match_num)

					elif ASSIGNMENT == "image":
						if self.overlap_truth(pnt[:2], rec, figure.shape) and self.overlap_truth(pnt[2:], rec, figure.shape):
							match_labels.append(match_num)
							match_labels.append("self_overlap") # if it contains "self_overlap", then it contains matching features within same single image

						elif self.overlap_truth(pnt[:2], rec, figure.shape) or self.overlap_truth(pnt[2:], rec, figure.shape):
							match_labels.append(match_num)

				if match_labels:
					self.Database["reference_label"][feature_i] = match_labels




	def overlap_truth(self, point, rect, shape):
		"""
		returns truth if point is contained within bounding box
		
		INPUTS:
		---------
		point: numpy array of size (2,)
			contains x,y coordinate of a match label
		rect: tuple (x,y,w,h)
			contains bounding box coordingate for segmented region
		shape: tuple
			shape of the figure

		RETURNS:
		---------
		1: if point is contained within bounding box
		0: otherwise
		"""
		background = np.zeros(shape[:2])

		background[np.maximum(rect[1]-25, 0):rect[1]+rect[3], np.maximum(rect[0]-25,0):rect[0]+rect[2]] = 1

		return background[(int(point[1]), int(point[0]))]


##
# View non-labeled matches
##

def Draw_rect(figure, b):
	cv2.rectangle(figure, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), color=(255,0,0), thickness=3)

def show_fig(figure):
	plt.figure()
	plt.imshow(figure)
	plt.show()

def view_sorted(data, score_type, Figures, Database):
	"""
	view the matches in descending scored order that do not have a ground-truth label.
	This is used to verify that the dataset is not miss-labeling true copies
	"""
	sort_indices = np.argsort(np.array(data[score_type]), axis=0)

	if score_type == "ratios":
		sort_indices = sort_indices[::-1]
	for key in data.keys():
		data[key] = np.array(data[key])[sort_indices]

	for i, pair in enumerate(np.array(data['pairs'])):
		fig_num = np.array(data["figure_number"])[i][0]
		y_truth = np.array(data["y_truth"])[i][0]

		figure = WesternBlot(dir_dictionary=Figures[fig_num], label="images", folder="eBik_output").figure.copy()
		if not y_truth:
			# first feature bounding box
			pair = pair[0]
			i_loc1 = Database["image_locs"][pair[0]]
			b_loc1 = Database["blot_locs"][pair[0]]
			b1 = get_blot_points(i_loc1, b_loc1)
			# second feature bounding box
			i_loc2 = Database["image_locs"][pair[1]]
			b_loc2 = Database["blot_locs"][pair[1]]
			print(i_loc2, b_loc2)
			b2 = get_blot_points(i_loc2, b_loc2)

		else:
			continue
		im = figure[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
		perc_white = np.sum(im == 255) / np.size(im)
		perc_black = np.sum(im == 0) / np.size(im)
		print("Percent (white, black: ", (perc_white, perc_black))
		Draw_rect(figure, b1)
		Draw_rect(figure, b2)

		show_fig(figure)



	
if __name__ == '__main__':
	Figures = np.load('testFigures/eBik_output/batches/eBik_testBatch.npy')
	Database = np.load('testFigures/eBik_output/fingerprints/batch_wb_clf_white2.npy').item()
	filename = "testFigures/eBik_output/reference_analysis_wb_clf_white2.npy"
	data = np.load(filename).item()

	view_sorted(data.copy(), "ratios", Figures, Database)
	#output_file = 'testFigures/eBik_output/annotations.npy' 
	# output_file = 'testFigures/eBik_output/annotations_images.npy' 

	# Annotation = AnnotateMatches(Figures, "eBik_output", output_file)
	# np.save(output_file, Annotation.match_points)

	# for Figure in Figures:
	# 	print(Figure["paper_idx"])

	# input_ = np.load(output_file).item()

	# print(len(input_["figure_number"]))


