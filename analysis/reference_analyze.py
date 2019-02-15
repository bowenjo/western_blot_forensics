import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import cv2 
import os

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
font = {'size'   : 10}
matplotlib.rc('font', **font)


def true_positive_rate(tp, total_pos):
	return (tp+1) / total_pos

def false_positive_rate(data, i, tp, total_pos, scale_fpr):
	if scale_fpr == "fig":
		total_neg = len(np.unique(data["figure_number"])) - total_pos
		fp = len(np.unique(data["figure_number"][:i+1])) - (tp+1)
	elif scale_fpr == "feature":
		fp = np.sum(1 - data["y_truth"][:i+1])
		tn = np.sum(1 - data["y_truth"][i+1:])
		total_neg = fp + tn

	return fp / total_neg

def roc_curve_ref(data, annotations, score_type, scale_tpr, scale_fpr):
	sort_indices = np.argsort(np.array(data[score_type]), axis=0)
	if score_type == "ratios":
		sort_indices = sort_indices[::-1]
	for key in data.keys():
		data[key] = np.array(data[key])[sort_indices]

	rec_match_ind = np.where(np.array(data["y_truth"]) == 1)

	if scale_tpr == "fig":
		total_pos = len(np.unique(annotations["figure_number"]))
		fig_recs = np.array(data["figure_number"])[rec_match_ind]
		u, u_i = np.unique(fig_recs, return_index = True)
		cut_points = rec_match_ind[0][u_i]
	elif scale_tpr == "feature":
		total_pos = len(annotations["match_num"]) 
		rec_match_nums = np.array(data["match_nums"])[rec_match_ind]
		u, u_i = np.unique([sublist[-1] for sublist in rec_match_nums], return_index=True)
		cut_points = rec_match_ind[0][u_i]
	elif scale_tpr == "image":
		rec_match_nums = np.array(data["match_nums"])[rec_match_ind]
		u, u_i = np.unique([sublist[-1] for sublist in rec_match_nums], return_index=True)
		cut_points = rec_match_ind[0][u_i]
		total_pos = 49 #len(cut_points)

	fpr = []; tpr = []
	for tp, i in enumerate(np.sort(cut_points)):
		fpr.append(false_positive_rate(data, i, tp, total_pos, scale_fpr))
		tpr.append(true_positive_rate(tp, total_pos))

	fpr.append(1)
	tpr.append(tpr[-1])

	return fpr, tpr, roc_auc(fpr,tpr), i/34042

def roc_auc(fpr, tpr):
    return np.sum(np.diff(np.array([0] + fpr)) * tpr)

def draw_roc(ax, fprs, tprs, colors, labels):
	cmap_gray = cm.get_cmap("gray")

	# ROC
	for i in range(len(labels)):
		ax.step(fprs[i], tprs[i], where='post', color=colors[i], lw=3, label= labels[i])
	
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.plot([0,1],[0,1], 'r--', lw=2)
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_xlim([0,1])
	ax.set_ylim([0,1.05])
	ax.yaxis.grid(which="major", color=cmap_gray(.8), linestyle='--', linewidth=1)
	ax.legend()

def draw_auc(ax, aucs, args, names):
	cmap_gray = cm.get_cmap("gray")

	# AUC
	N = len(args)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.15      # the width of the bars

	for auc, name in zip(aucs, names): 
		ax.plot(ind, auc, markersize=10, marker='s', lw=3, label=name)

	#ax.set_ylim([.5,1.05])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.grid(which="major", color=cmap_gray(.8), linestyle='--', linewidth=1)
	ax.set_axisbelow(True)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('AUC score')
	ax.set_xticks(ind)
	ax.set_xticklabels(args)
	ax.legend()

def roc_plot_features(data, annotations):
	# compares feature-wise distance and distance ratio scores rescaled for 3 axes
	colors = ['b', 'g']
	labels = ['Distance Score', 'Distance Ratio Score']

	fig, ax = plt.subplots(1,2, figsize=(9,4))

	fpr_r_feat, tpr_r_feat, auc_r_feat = roc_curve_ref(data.copy(), annotations, "ratios", "feature", "feature")
	fpr_d_feat, tpr_d_feat, auc_d_feat = roc_curve_ref(data.copy(),annotations, "distances", "feature", "feature")
	fpr_feat_feat = [fpr_d_feat, fpr_r_feat]
	tpr_feat_feat = [tpr_d_feat, tpr_r_feat]
	draw_roc(ax[0], fpr_feat_feat, tpr_feat_feat, colors, labels)
	ax[0].set_title("All feature matches")

	fpr_r_fig, tpr_r_fig, auc_r_fig = roc_curve_ref(data.copy(), annotations, "ratios", "fig", "feature")
	fpr_d_fig, tpr_d_fig, auc_d_fig = roc_curve_ref(data.copy(), annotations, "distances", "fig", "feature")
	fpr_feat_fig = [fpr_d_fig, fpr_r_fig]
	tpr_feat_fig = [tpr_d_fig, tpr_r_fig]
	draw_roc(ax[1], fpr_feat_fig, tpr_feat_fig, colors, labels)
	ax[1].set_title("At least one match per figure")

	# fpr_r_fig_fig, tpr_r_fig_fig, auc_r_fig_fig = roc_curve_ref(data.copy(), annotations, "ratios", "fig", "fig")
	# fpr_d_fig_fig, tpr_d_fig_fig, auc_d_fig_fig = roc_curve_ref(data.copy(), annotations, "distances", "fig", "fig")
	# fpr_fig_fig = [fpr_d_fig_fig, fpr_r_fig_fig]
	# tpr_fig_fig = [tpr_d_fig_fig, tpr_r_fig_fig]
	# draw_roc(ax[2], fpr_fig_fig, tpr_fig_fig, colors, labels)
	# ax[2].set_title("figures alo fig")
	
	#plt.savefig('Projects/feature_test_eBik/vis/feature_ref_vis.png')
	#plt.show()

def roc_plot_image(data, data_sift, annotations):
	# compares image-wise distance ratio scores rescaled for 3 axes for blotcutter and sift
	colors = ['g', 'orange']
	labels = ['BlotCutter', 'SIFT']

	fig, ax = plt.subplots(1,2, figsize=(9,4))

	# fpr_feat, tpr_feat, _ = roc_curve(data["y_truth"], data["ratios"])
	# fpr_s_feat, tpr_s_feat, _ = roc_curve(data_sift["y_truth"], data_sift["ratios"])

	fpr_feat, tpr_feat, auc_feat = roc_curve_ref(data.copy(), annotations, "ratios", "image", "feature")
	fpr_s_feat, tpr_s_feat, auc_s_feat = roc_curve_ref(data_sift.copy(),annotations, "distances", "image", "feature")
	fpr_feat_feat = [fpr_feat, fpr_s_feat]
	tpr_feat_feat = [tpr_feat, tpr_s_feat]
	draw_roc(ax[0], fpr_feat_feat, tpr_feat_feat, colors, labels)
	ax[0].set_title("All image matches")

	fpr_fig, tpr_fig, auc_fig = roc_curve_ref(data.copy(), annotations, "ratios", "fig", "feature")
	fpr_s_fig, tpr_s_fig, auc_s_fig = roc_curve_ref(data_sift.copy(), annotations, "ratios", "fig", "feature")

	fpr_feat_fig = [fpr_fig, fpr_s_fig]
	tpr_feat_fig = [tpr_fig, tpr_s_fig]
	draw_roc(ax[1], fpr_feat_fig, tpr_feat_fig, colors, labels)
	ax[1].set_title("At least one match per figure")

	# fpr_fig_fig, tpr_fig_fig, auc_fig_fig = roc_curve_ref(data.copy(), annotations, "ratios", "fig", "fig")
	# fpr_s_fig_fig, tpr_s_fig_fig, auc_s_fig_fig = roc_curve_ref(data_sift.copy(), annotations, "ratios", "fig", "fig")
	# fpr_fig_fig = [fpr_fig_fig, fpr_s_fig_fig]
	# tpr_fig_fig = [tpr_fig_fig, tpr_s_fig_fig]
	# draw_roc(ax[2], fpr_fig_fig, tpr_fig_fig, colors, labels)
	# ax[2].set_title("figures a.l.o. fig")
	
	#plt.savefig('Projects/image_test_eBik/vis/image_ref_vis.png')
	#plt.show()




if __name__ == "__main__":
	dir_ = os.path.expanduser("~") + "/Documents/GitHub/NonRepositories/westernBlotForensics/wb_forensics/"

	data_filename_feat = "Projects/feature_test_eBik/saves/reference_analysis.npy"
	data_filename_image = "Projects/image_test_eBik/saves/reference_analysis.npy"
	data_filename_sift = "Projects/sift_test_eBik/saves/reference_analysis.npy"
	annotations_filename = "Datasets/eBik/annotations.npy"

	data_feat = np.load(dir_ + data_filename_feat).item()
	data_image = np.load(dir_ + data_filename_image).item()
	data_sift = np.load(dir_ + data_filename_sift).item()
	annotations = np.load(dir_ + annotations_filename).item()

	roc_plot_features(data_feat, annotations)
	roc_plot_image(data_image, data_sift, annotations)
	
	# rec_match_ind = np.where(np.array(data_sift["y_truth"]) == 1)
	# rec_match_nums = np.array(data_sift["match_nums"])[rec_match_ind]

	# print(len(rec_match_nums))
	# count = []
	# for sublist in rec_match_nums:
	# 	for m in sublist:
	# 		if m != 'self_overlap':
	# 			count.append(m)

	# print(len(np.unique(count)))