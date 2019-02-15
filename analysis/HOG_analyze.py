import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import cv2 
import time

from wbFingerprint import WesternBlot 
from wbForensicsHOG import wbForensicsHOG
from sklearn.metrics import roc_curve, roc_auc_score
from utils.wbFingerprintUtils import print_time

from analysis.reference_analyze import draw_roc, draw_auc

class HOG_analyzer(object):
	def __init__(self, data):
		self.data = data

		self.Database = {"feature_vectors":[],
						 "label": [],
						 "figure_number":None}
	def run(self):
		self.collect_fingerprints()
		Forensics = wbForensicsHOG(self.Database)
		Forensics.KDTree_pairs(len(self.Database["feature_vectors"])+1)

		dists = Forensics.dists[0]
		ratios = Forensics.ratios[0]

		match_labels = np.array(self.Database["label"])[Forensics.pairs[0]]
		truth = np.where(np.diff(match_labels, axis=1) == 0)
		false = np.where(np.diff(match_labels, axis=1) != 0)
		y_truth = np.zeros_like(dists)
		y_truth[truth] = 1
		y_truth[false] = 0

		return y_truth, np.array(dists), np.array(ratios)

	def collect_fingerprints(self):
		i = 0
		for img, mark in zip(self.data["image"], self.data["mark"]):
			des = self.calc_fingerprint(img, mark, start_idx = 1)
			if des is not None:
				self.Database["feature_vectors"].append(des)
				self.Database["label"].append(self.data["label"][i])
			i+=1	

		shape = (len(self.Database["label"]) , self.Database["feature_vectors"][-1].shape[1])
		self.Database["feature_vectors"] = np.array(self.Database["feature_vectors"]).reshape(shape)		
		self.Database["figure_number"] = [0]*self.Database["feature_vectors"].shape[0]

	def calc_fingerprint(self, img, mark, start_idx):
		WB = WesternBlot()
		fingerprint, _ = WB.appendFingerprintData(img, mark, start_idx)
		if len(fingerprint["descriptor"]) == 0:
			return None
		else:
			assert len(fingerprint["descriptor"]) == 1
			return fingerprint["descriptor"]


def plot_AUC_synthetic(data, args, title, labels, save_file, pre_load=False):
	if pre_load:
		pre_load_dict = np.load(save_file).item()
		AUC = pre_load_dict["AUC_Dists"]
		AUC_ratios = pre_load_dict["AUC_Ratios"]
	else:	
		i = 0
		AUC = []; AUC_ratios = []
		for d, arg in zip(data, args):
			analyzer = HOG_analyzer(d)
			y_truth, dists, ratios = analyzer.run()

			AUC.append(roc_auc_score(y_truth, np.max(dists) - dists))
			AUC_ratios.append(roc_auc_score(y_truth, ratios))
			i+=1
			print(print_time(time.time()), ": {} of {} transformation arguments completed".format(i, len(args)))
		np.save(save_file, {"AUC_Dists": AUC, "AUC_Ratios": AUC_ratios})


def draw_auc(ax, AUC, AUC_ratios, labels, title, x_label):	
	# Draw plot
	N = len(AUC)

	ind = np.arange(N)  # the x locations for the groups
	width = .20      # the width of the bars

	cmap_blue = cm.get_cmap("Blues")
	cmap_green = cm.get_cmap("Greens")
	cmap_gray = cm.get_cmap("gray")

	rects1 = ax.bar(ind, AUC, width, color=cmap_blue(.59), label='L2 Distance')
	rects2 = ax.bar(ind + width, AUC_ratios, width, color=cmap_green(.59), label='L2 Distance Ratio')
	# rects1 = ax.plot(ind, AUC, markersize=10, lw=3, color=cmap_blue(.59), label='L2 Distance', marker='s')
	# rects2 = ax.plot(ind, AUC_ratios, markersize=10, lw=3, color=cmap_green(.59), label='L2 Distance Ratio', marker = 's')

	ax.set_ylim([.5,1.05])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.grid(which="major", color=cmap_gray(.8), linestyle='--', linewidth=1)
	ax.set_axisbelow(True)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('AUC score')
	ax.set_xlabel(x_label)
	ax.set_title(title)
	ax.set_xticks(ind + (width/2))
	ax.set_xticklabels(labels)

