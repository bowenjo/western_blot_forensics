import numpy as np
import matplotlib.pyplot as plt
import wbFingerprintUtils as utils
import wbFingerprint as wb
import itertools
import os
import cv2

"""
Collection of plotting functions to present results of different descriptor types on performance and implementation at detecting duplicated blots.
"""

class BlotPerformance(object):
	"""
	Class of plotting functions to present results of different descriptor types on performance and implementation of detecting duplicated blots.
	Looks at three descriptor types: Histogram of Oriented gradients, image moments (HU), and raw-pixel cross correlation. Compares as a function 
	different transformation types: rotating, scaling, and brightness/contrast adjustments
	"""

	def __init__(self, rotate_dir=None, scale_dir=None, adjLinCon_dir=None, adjNonLinCon_dir=None):

		if rotate_dir is not None:
			self.rotationDicts = []
			self.load_transform_dict(self.rotationDicts, rotate_dir)

		if scale_dir is not None:
			self.scaleDicts = []
			self.load_transform_dict(self.scaleDicts, scale_dir)

		if adjLinCon_dir is not None:
			self.adjLinConDicts = []
			self.load_transform_dict(self.adjLinConDicts, adjLinCon_dir)

		if adjNonLinCon_dir is not None:
			self.adjNonLinConDicts = []
			self.load_transform_dict(self.adjNonLinConDicts, adjNonLinCon_dir)

	def load_transform_dict(self, transform_dict, directory):
		for subdir, dirs, files in os.walk(directory):
			for file in files:
				dictionary = np.load(os.path.join(directory, file)).item()
				transform_dict.append(dictionary)

	def get_params(self, dictionary_type):

		if dictionary_type == "rotation":
			transform_dict = self.rotationDicts
			xlabel = "Rotation Angle (Degrees)"
			label_type = "rotation" 
			show_title = "theta"
			num_subplts = 1
			width = 1
		elif dictionary_type == "scale":
			transform_dict = self.scaleDicts
			xlabel = "Scale"
			label_type = "scaling"
			show_title = "scaleFactor"
			num_subplts = 1
			width = .05
		elif dictionary_type == "linear-contrast":
			transform_dict = self.adjLinConDicts
			xlabel = "Linear Contrast"
			label_type = "lnr. con. adj."
			show_title = "contrastFactor"
			num_subplts = 2
			width = .05
		elif dictionary_type in ["non-linear-contrast-S2S_INV", 'non-linear-contrast-LOW2HIGH', 'non-linear-contrast-ML2EX', 'non-linear-contrast-random']:
			transform_dict = self.adjNonLinConDicts
			xlabel = "Distance From Zero Contrast Change"
			label_type = "nlnr. con. adj."
			show_title = "distance"
			num_subplts = 3
			width = 5
		else:
			raise RuntimeError(dictionary_type + " is not a recognized dictionary type")

		return(transform_dict, xlabel, label_type, show_title, num_subplts, width)

	def showExample(self, bool, dictionary_type, transform_dictionary, show_title, num_subplts, figsize):

		if bool is True:
			exp_blots = transform_dictionary[7]
			exp_fig, exp_ax = plt.subplots(num_subplts,len(exp_blots["image"]), figsize=figsize)

			if dictionary_type == "linear-contrast":
				for i, a in enumerate(exp_ax[0]):
					a.imshow(exp_blots["image"][i], "gray")
					a.set_title(str(exp_blots[show_title][i]))
					a.axis("off")
				for i, a in enumerate(exp_ax[1]):
					a.hist(cv2.cvtColor(exp_blots["image"][i], cv2.COLOR_BGR2GRAY).flatten(), 50, [0,256])
					a.set_xlim(0,255)
					a.axis("off")
			elif dictionary_type == "non-linear-contrast-S2S_INV" or dictionary_type == "non-linear-contrast-random":
				for i, a in enumerate(exp_ax[0]):
					a.imshow(exp_blots["image"][i], "gray")
					a.axis("off")
				for i, a in enumerate(exp_ax[1]):
					a.hist(cv2.cvtColor(exp_blots["image"][i], cv2.COLOR_BGR2GRAY).flatten(), 50, [0,256])
					a.set_xlim(0,255)
					a.axis("off")
				for i, a in enumerate(exp_ax[2]):
					a.plot(exp_blots["xdata"], exp_blots["ydata"][i], ":", color="r")
					a.axis("off")
			elif dictionary_type == 'non-linear-contrast-LOW2HIGH' or dictionary_type == 'non-linear-contrast-ML2EX':
				for i, a in enumerate(exp_ax[0]):
					a.imshow(exp_blots["image"][i], "gray")
					a.axis("off")
				for i, a in enumerate(exp_ax[1]):
					a.hist(cv2.cvtColor(exp_blots["image"][i], cv2.COLOR_BGR2GRAY).flatten(), 50, [0,256])
					a.set_xlim(0,255)
					a.axis("off")
				for i, a in enumerate(exp_ax[2]):
					a.plot(exp_blots["xdata"][i], exp_blots["ydata"][i], ":", color="r")
					a.axis("off")
			else:
				for i, a in enumerate(exp_ax):
					a.imshow(exp_blots["image"][i], "gray")
					a.set_title(str(exp_blots[show_title][i]))
					a.axis("off")

	def computeResults(self, transform_dict, show_title, EQUALIZE):

		x_values = [] # intialize x-values array for plot
		y_values_des = [] # intitialize y-values array for plot
		y_values_ncc = []
		control_dist_des = [] # intialize control distribution to compare against variantion across images
		control_dist_ncc = []

		N = len(transform_dict)

		for dictionary in transform_dict: # loop through each image-scale dictionary
			x_factor = dictionary[show_title]
			blots = dictionary["image"]
			image_og = dictionary["original"]

			## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
			WB_og = wb.WesternBlot()
			WB_og.figure = image_og
			WB_og.figure_gray = cv2.cvtColor(image_og, cv2.COLOR_BGR2GRAY)
			WB_og.westernBlotExtractor()
			WB_og.blotExtractorWatershed(255, Visualize=False, EQUALIZE=EQUALIZE)

			des_og = WB_og.fingerprint["descriptor"][0]
			ncc_og = WB_og.fingerprint["blots"][0]

			distances_des = []
			distances_ncc = []
			for blot in blots:
				## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
				WB = wb.WesternBlot()
				WB.figure = blot
				WB.figure_gray = cv2.cvtColor(blot, cv2.COLOR_BGR2GRAY)
				WB.westernBlotExtractor()
				WB.blotExtractorWatershed(255, Visualize=False, EQUALIZE=EQUALIZE)
		
				des = WB.fingerprint["descriptor"][0]
				ncc = WB.fingerprint["blots"][0]
					
				distances_des.append(np.linalg.norm(des - des_og)) # compute L2-distance between original image and rotated form
				if ncc_og.size >= ncc.size or ncc_og.size < ncc.size:
					ncc_og_scaled = cv2.resize(ncc_og, (ncc.shape[1], ncc.shape[0]), interpolation=cv2.INTER_LINEAR)
					distances_ncc.append(np.max(cv2.matchTemplate(ncc_og_scaled, ncc, cv2.TM_CCOEFF_NORMED)))

			x_values.append(x_factor) # append each corresponding angle row to x-value array
			y_values_des.append(distances_des) # append each rotated image row to y-value array
			y_values_ncc.append(distances_ncc)
			control_dist_des.append(des_og) # append each original image descriptor for the control line
			control_dist_ncc.append(ncc_og)

		## common x_values for each plot
		x_values = np.mean(np.array(x_values), axis=0) # corresponding angles 

		## ------------------------------------------------------------------------------------------------

		## solve for the control-value line (HOG)
		# TODO Put in error if len(self.control_dist) is less than or equal to one
		control_dist_des = list(itertools.combinations(control_dist_des, 2))
		control_des = [np.linalg.norm(np.diff(c)) for c in control_dist_des]
		control_des_mean = np.mean(control_des)

		## first plot results (HOG)
		y_values_mean_des = np.mean(np.array(y_values_des), axis=0) # average over images
		y_values_std_des = np.std(np.array(y_values_des), axis=0) #/ np.sqrt(np.array(y_values_des).shape[0])
		
		## second plot results (HOG)
		y_values2_mean_des = (control_des_mean - y_values_mean_des) / control_des_mean
		y_values2_std_des = (control_des_mean - (y_values_mean_des-y_values_std_des)) / control_des_mean
		y_values2_std_des = y_values2_mean_des - y_values2_std_des

		## -----------------------------------------------------------------------------------------------

		## solve for control-value line (NCC)
		control_dist_ncc = list(itertools.combinations(control_dist_ncc, 2))
		control_ncc = []
		for c in control_dist_ncc:
			if c[0].size >= c[1].size:
				c_scaled = cv2.resize(c[0], (c[1].shape[1], c[1].shape[0]), interpolation=cv2.INTER_LINEAR)
				control_ncc.append(np.max(cv2.matchTemplate(c_scaled, c[1], cv2.TM_CCOEFF_NORMED)))
			else:
				c_scaled = cv2.resize(c[1], (c[0].shape[1], c[0].shape[0]), interpolation=cv2.INTER_LINEAR)
				control_ncc.append(np.max(cv2.matchTemplate(c_scaled, c[0], cv2.TM_CCOEFF_NORMED)))

		control_ncc_mean = np.mean(control_ncc)

		## first plot resilts (NCC)
		y_values_mean_ncc = np.mean(np.array(y_values_ncc), axis=0)
		y_values_std_ncc = np.std(np.array(y_values_ncc), axis=0) #/ np.sqrt(np.array(y_values_ncc).shape[0])

		## second plot results (NCC)
		y_values2_mean_ncc = -(control_ncc_mean - y_values_mean_ncc) / (1 - control_ncc_mean + np.finfo(float).eps)
		y_values2_std_ncc = np.abs(control_ncc_mean - np.abs(y_values_mean_ncc - y_values_std_ncc)) / (1 - control_ncc_mean + np.finfo(float).eps)
		y_values2_std_ncc = y_values2_mean_ncc - y_values2_std_ncc

		## -------------------------------------------------------------------------------------------------

		return((x_values,
			   y_values_mean_des,
			   y_values_std_des,
			   y_values2_mean_des,
			   y_values2_std_des,
			   y_values_mean_ncc,
			   y_values_std_ncc,
			   y_values2_mean_ncc,
			   y_values2_std_ncc,
			   control_des,
			   control_ncc,
			   N))

	def plotResults(self, results, y1lim, y2lim, xlabel, label_type, width, figsize, title):
		x, y1, y1_err, y11, y11_err, y2, y2_err, y22, y22_err, control1, control2, N = results
		
		## plot results
		f, a = plt.subplots(2,2,figsize = figsize)

		f.suptitle(title + " (N = "+str(N)+")", fontsize=16)

		## ------------------------------------------------------------------------------------------------------------------------------------------------

		## First Plot (HOG)
		a[0,0].errorbar(x, y1, yerr = y1_err, fmt='o', label="Duplicated blots")
		a[0,0].set_xlabel(xlabel)
		a[0,0].set_ylabel("Mean Euclidean Distance")
		a[0,0].set_ylim(0, 1.6)
		a[0,0].plot((np.min(x),np.max(x)), (np.mean(control1), np.mean(control1)), "r-", label = "Control")
		a[0,0].plot((np.min(x),np.max(x)), (np.min(control1), np.min(control1)), "r--")
		a[0,0].plot((np.min(x),np.max(x)), (np.max(control1), np.max(control1)), "r--")
		a[0,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.) 

		## Second Plot (HOG)
		a[0,1].errorbar(x, y11, fmt = 'o', yerr = y11_err)
		a[0,1].set_xlabel(xlabel)
		a[0,1].set_ylabel("Similarity")
		a[0,1].set_ylim(-1.05,1.05)
		a[0,1].plot((np.min(x),np.max(x)), (0,0), "k--")
		#a[0,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

		## -------------------------------------------------------------------------------------------------------------------------------------------------

		## First Plot (NCC)
		a[1,0].errorbar(x, y2, yerr = y2_err,  fmt='o', label="Duplicated blots", color='orange', ecolor='orange')
		a[1,0].set_xlabel(xlabel)
		a[1,0].set_ylabel("NCC Coefficient")
		a[1,0].set_ylim(0, 1.1)
		a[1,0].plot((np.min(x),np.max(x)), (np.mean(control2),np.mean(control2)), "r-", label = "Control")
		a[1,0].plot((np.min(x),np.max(x)), (np.min(control2), np.min(control2)), "r--")
		a[1,0].plot((np.min(x),np.max(x)), (np.max(control2), np.max(control2)), "r--")
		a[1,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

		## Second Plot (NCC)
		a[1,1].errorbar(x, y22, fmt = 'o', yerr = y22_err, ecolor='orange', color='orange')
		a[1,1].set_ylabel("Similarity")
		a[1,1].set_xlabel(xlabel)
		a[1,1].set_ylim(-1.05,1.05)
		a[1,1].plot((np.min(x),np.max(x)), (0,0), "k--")
		#a[1,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

		plt.gcf().subplots_adjust(top=.9, bottom=.1, hspace=.35, wspace=.3)

	def plotResultsContrastNL(self, results, xlabel, label_type, figsize, title, CUSTOM):
		x, y1, y1_err, y11, y11_err, y2, y2_err, y22, y22_err, control1, control2, N = results

		# plot results
		f, a = plt.subplots(2,2, figsize=figsize)

		f.suptitle(title + " (N = "+str(N)+")", fontsize=16, y=1)

		if CUSTOM == 'S2S_INV':
			label1 = "S-Shaped"
			label2 = "Inverted-S-Shaped"
		elif CUSTOM == "LOW2HIGH":
			label1 = "HIGH-Saturation"
			label2 = "LOW-Saturation"
		elif CUSTOM == "ML2EX":
			label1 = "Mid-Levels"
			label2 = "Extremes"

		## First Plot (HOG)
		a[0,0].errorbar(x, y1, yerr = y1_err, fmt='o') # label="Duplicated blots (with "+str(label_type)+")")
		if CUSTOM == 'S2S_INV' or CUSTOM == 'LOW2HIGH' or CUSTOM == 'ML2EX':
			a[0,0].plot(x[:(int(len(x)/2)+1)], y1[:(int(len(x)/2)+1)], c = 'blue', label=label1)
			a[0,0].plot(x[int(len(x)/2):], y1[int(len(x)/2):], c = 'green', label=label2)
			a[0,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
		a[0,0].set_xlabel(xlabel)
		a[0,0].set_ylabel("Mean Euclidean Distance")
		a[0,0].set_ylim(0, 1.6)
		a[0,0].plot((np.min(x),np.max(x)), (np.mean(control1), np.mean(control1)), "r-", label = "Control")
		a[0,0].plot((np.min(x),np.max(x)), (np.min(control1), np.min(control1)), "r--")
		a[0,0].plot((np.min(x),np.max(x)), (np.max(control1), np.max(control1)), "r--")

		## Second Plot (HOG)
		a[0,1].errorbar(x, y11, fmt='o')
		if CUSTOM == 'S2S_INV' or CUSTOM == 'LOW2HIGH' or CUSTOM == 'ML2EX':
			a[0,1].plot(x[:(int(len(x)/2)+1)], y11[:(int(len(x)/2)+1)], c = 'blue', label=label1)
			a[0,1].plot(x[int(len(x)/2):], y11[int(len(x)/2):], c = 'green', label=label2)
			a[0,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.) 
		a[0,1].set_xlabel(xlabel)
		a[0,1].set_ylabel("Similarity")
		a[0,1].set_ylim(-1.05,1.05)
		a[0,1].plot((np.min(x),np.max(x)), (0,0), "k--")


		## ----------------------------------------------------------------------------------------------------------------------

		## First Plot (NCC)
		a[1,0].errorbar(x, y2, fmt='o', ecolor="orange", color="orange", yerr=y2_err)
		if CUSTOM == 'S2S_INV' or CUSTOM == 'LOW2HIGH' or CUSTOM == 'ML2EX':
			a[1,0].plot(x[:(int(len(x)/2)+1)], y2[:(int(len(x)/2)+1)], c = 'blue', label=label1)
			a[1,0].plot(x[int(len(x)/2):], y2[int(len(x)/2):], c = 'green', label=label2)
			a[1,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
		a[1,0].set_xlabel(xlabel)
		a[1,0].set_ylabel("NCC Coefficient")
		a[1,0].set_ylim(0, 1.05)
		a[1,0].plot((np.min(x),np.max(x)), (np.mean(control2),np.mean(control2)), "r-", label = "Control")
		a[1,0].plot((np.min(x),np.max(x)), (np.min(control2), np.min(control2)), "r--")
		a[1,0].plot((np.min(x),np.max(x)), (np.max(control2), np.max(control2)), "r--")

		## Second Plot (NCC)
		a[1,1].errorbar(x, y22, fmt='o', ecolor='orange', color='orange')
		if CUSTOM == 'S2S_INV' or CUSTOM == 'LOW2HIGH' or CUSTOM == 'ML2EX':
			a[1,1].plot(x[:(int(len(x)/2)+1)], y22[:(int(len(x)/2)+1)], c = 'blue', label=label1)
			a[1,1].plot(x[int(len(x)/2):], y22[int(len(x)/2):], c = 'green', label=label2)
			a[1,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
		a[1,1].set_xlabel(xlabel)
		a[1,1].set_ylabel("Similarity")
		a[1,1].set_ylim(-1.05,1.05)
		a[1,1].plot((np.min(x),np.max(x)), (0,0), "k--")

		plt.gcf().subplots_adjust(top=.9, bottom=.1, hspace=.35, wspace=.3)

		#a[1,1].legend(loc="upper left", prop={'size':9}) 

	def plotPerformance(self, dictionary_type, figsize, title, show_example=False, EQUALIZE=False):
		"""
		plots performance of each descriptor method as a function of rotation angle. Performance is measured by L2 euclidean distance between original
		blot and copied blot of given rotation angle.

		Parameters:
		------------
		figsize: tuple (x-dimension, y-dimension)
			sets the figure size
		title: str
			sets title of figure
		show_eaxample: boolean (default: False)
			If True, plots an example of one blot at each rotation angle

		"""
		
		transform_dict, xlabel, label_type, show_title, num_subplts, width = self.get_params(dictionary_type)

		## show example 
		self.showExample(bool=show_example,
						 dictionary_type = dictionary_type, 
						 transform_dictionary = transform_dict,
						 show_title = show_title,
						 num_subplts = num_subplts, 
						 figsize = (25,3))

		# compute results
		results = self.computeResults(transform_dict = transform_dict,
					   				  show_title = show_title,
					   				  EQUALIZE=EQUALIZE)


		if dictionary_type in ["non-linear-contrast-S2S_INV", 'non-linear-contrast-LOW2HIGH', 'non-linear-contrast-ML2EX', 'non-linear-contrast-random']:
			if dictionary_type == 'non-linear-contrast-S2S_INV':
				CUSTOM = 'S2S_INV' 
			elif dictionary_type == 'non-linear-contrast-LOW2HIGH':
				CUSTOM = 'LOW2HIGH' 
			elif dictionary_type == "non-linear-contrast-ML2EX":
				CUSTOM = 'ML2EX' 
			elif dictionary_type == "non-linear-contrast-random":
				CUSTOM = 'random' 
			self.plotResultsContrastNL(results,
									   xlabel = xlabel,
									   label_type = label_type,
									   figsize=figsize,
									   title=title,
									   CUSTOM=CUSTOM)
		else:
			self.plotResults(results,
							 y1lim = 1.15, 
							 y2lim = 1.15,
						 	 xlabel = xlabel, 
							 label_type = label_type,
							 width = width, 
							 figsize = figsize, 
							 title = title)













		

