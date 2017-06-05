import numpy as np
import matplotlib.pyplot as plt
import wbFingerprintUtils as utils
import wbMomentsFingerprint as wb
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

	def __init__(self, rotate_dir=None, scale_dir=None, adjCon_dir=None):

		if rotate_dir is not None:
			self.rotationDicts = []
			for subdir, dirs, files in os.walk(rotate_dir):
				for file in files: 
					rot_dic = np.load(os.path.join(rotate_dir, file)).item()
					self.rotationDicts.append(rot_dic)

		if scale_dir is not None:
			self.scaleDicts = []
			for subdir, dirs, files in os.walk(scale_dir):
				for file in files:
					scale_dic = np.load(os.path.join(scale_dir, file)).item()
					self.scaleDicts.append(scale_dic)

		if adjCon_dir is not None:
			self.adjConDicts = []
			for subdir, dirs, files in os.walk(adjCon_dir):
				for file in files:
					adjCon_dic = np.load(os.path.join(adjCon_dir, file)).item()
					self.adjConDicts.append(adjCon_dic)



	def rotationPerformance(self, figsize, title, show_example=False):
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
		## Plot example figure of rotated blots

		if show_example is True:
			exp_blots = self.rotationDicts[0]
			exp_fig, exp_ax = plt.subplots(1,len(exp_blots["image"]), figsize=(10,2))

			for i, a in enumerate(exp_ax):
				a.imshow(exp_blots["image"][i], "gray")
				a.set_title(str(exp_blots["theta"][i]))
				a.axis("off")


		## compute euclidean distances as a function of rotation angle:

		y_values_des = [] # intitialize y-values array for plot
		y_values_ncc = []
		x_values = [] # intialize x-values array for plot
		control_dist_des = [] # intialize control distribution to compare against variantion across images
		control_dist_ncc = []

		for r_dict in self.rotationDicts: # loop through each image-rotation dictionary 
			theta = r_dict["theta"] # list of angles
			blots_rot = r_dict["image"] # corresponding rotated image to theta
			image_og = r_dict["original"] # non-rotated image

			## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
			WB_og = wb.WesternBlot()
			WB_og.figure = image_og
			WB_og.figure_gray = cv2.cvtColor(image_og, cv2.COLOR_BGR2GRAY)
			WB_og.westernBlotExtractor()
			WB_og.blotExtractorWatershed(255, Visualize=False)

			des_og = WB_og.fingerprint["descriptor"][0]
			ncc_og = WB_og.fingerprint["blots"][0]
			
			distances_des = [] # initialize L2-distances array
			distances_ncc = []
			for blot in blots_rot: # loop through each rotated image
				## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
				WB = wb.WesternBlot()
				WB.figure = blot
				WB.figure_gray = cv2.cvtColor(blot, cv2.COLOR_BGR2GRAY)
				WB.westernBlotExtractor()
				WB.blotExtractorWatershed(255, Visualize=False)

				des = WB.fingerprint["descriptor"][0]
				ncc = WB.fingerprint["blots"][0]

				distances_des.append(np.linalg.norm(des - des_og, ord=2)) # compute L2-distance between original image and rotated form
				if ncc_og.size >= ncc.size or ncc_og.size < ncc.size:
					ncc_og_scaled = cv2.resize(ncc_og, (ncc.shape[1], ncc.shape[0]), interpolation=cv2.INTER_LINEAR)
					distances_ncc.append(np.max(cv2.matchTemplate(ncc_og_scaled, ncc, cv2.TM_CCOEFF_NORMED)))
				# else:
				# 	ncc_scaled = cv2.resize(ncc, (ncc_og.shape[1], ncc.shape[0]))
				# 	distances_ncc.append(np.max(cv2.matchTemplate(ncc_og, ncc_scaled, cv2.TM_CCOEFF_NORMED)))

			
			x_values.append(theta) # append each corresponding angle row to x-value array
			y_values_des.append(distances_des) # append each rotated image row to y-value array
			y_values_ncc.append(distances_ncc)
			control_dist_des.append(des_og) # append each original image descriptor for the control line
			control_dist_ncc.append(ncc_og)

		## common x_values for each plot
		x_values = np.mean(np.array(x_values), axis=0) # corresponding angles 

		## -------------------------------------------------------------------------------------------------------
		
		## solve for the control-value line (HOG)
		# TODO Put in error if len(self.control_dist) is less than or equal to one
		control_dist_des = list(itertools.combinations(control_dist_des, 2))
		control_des = np.mean([np.linalg.norm(np.diff(c)) for c in control_dist_des])
		control_max_des= np.max([np.linalg.norm(np.diff(c)) for c in control_dist_des])
		control_min_des= np.min([np.linalg.norm(np.diff(c)) for c in control_dist_des])

		## first plot results (HOG)
		y_values_mean_des = np.mean(np.array(y_values_des), axis=0) # average over images
		y_values_std_des = np.std(np.array(y_values_des), axis=0) / np.sqrt(np.array(y_values_des).shape[0])
		
		## second plot results (HOG)
		y_values2_mean_des = (control_des - y_values_mean_des) / control_des
		y_values2_std_des = (control_des - (y_values_mean_des-y_values_std_des)) / control_des
		y_values2_std_des = y_values2_mean_des - y_values2_std_des

		## --------------------------------------------------------------------------------------------------------

		## solve for control-value line (NCC)
		control_dist_ncc = list(itertools.combinations(control_dist_ncc, 2))
		control_ncc = []
		for c in control_dist_ncc:
			if c[0].size >= c[1].size:
				c_scaled = cv2.resize(c[0], (c[1].shape[1], c[1].shape[0]))
				control_ncc.append(np.max(cv2.matchTemplate(c_scaled, c[1], cv2.TM_CCOEFF_NORMED)))
			else:
				c_scaled = cv2.resize(c[1], (c[0].shape[1], c[0].shape[0]))
				control_ncc.append(np.max(cv2.matchTemplate(c_scaled, c[0], cv2.TM_CCOEFF_NORMED)))
		control_max_ncc = np.max(control_ncc)
		control_min_ncc = np.min(control_ncc)
		control_ncc = np.mean(control_ncc)

		## first plot resilts (NCC)
		y_values_mean_ncc = np.mean(np.array(y_values_ncc), axis=0)
		y_values_std_ncc = np.std(np.array(y_values_ncc), axis=0) / np.sqrt(np.array(y_values_ncc).shape[0])

		## second plot results (NCC)
		y_values2_mean_ncc = -(control_ncc - y_values_mean_ncc) / (1 - control_ncc + np.finfo(float).eps)
		y_values2_std_ncc = np.abs(control_ncc - np.abs(y_values_mean_ncc - y_values_std_ncc)) / (1 - control_ncc + np.finfo(float).eps)
		y_values2_std_ncc = y_values2_mean_ncc - y_values2_std_ncc

		## -------------------------------------------------------------------------------------------------------

		## plot results
		f, a = plt.subplots(2,2,figsize = figsize)

		f.suptitle(title + " (N = "+str(len(y_values_des))+")", fontsize=18)

		## ------------------------------------------------------------------------------------------------------------------------------------------------

		## First Plot (HOG)
		a[0,0].errorbar(x_values, y_values_mean_des, yerr = y_values_std_des, fmt='-o', label="Duplicated blots (with rotation)")
		a[0,0].set_xlabel("Rotation Angle (degrees)")
		a[0,0].set_ylabel("Mean Euclidean Distance")
		a[0,0].set_ylim(0, control_des+.65)
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_des,control_des), "r-", label = "Non-duplicated blots (without rotation)")
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_min_des, control_min_des), "r--")
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_max_des, control_max_des), "r--")
		a[0,0].legend(loc="upper left", prop={'size':9})

		## Second Plot (HOG)
		a[0,1].bar(x_values, y_values2_mean_des, 1, yerr = y_values2_std_des)
		a[0,1].set_xlabel("Rotation Angle (degrees)")
		a[0,1].set_ylabel("Similarity")
		a[0,1].set_ylim(0,1.15)
		a[0,1].plot((np.min(x_values),np.max(x_values)), (1,1), "r--", label="Zero Distance")
		a[0,1].legend(loc="upper left", prop={'size':9})  

		## -------------------------------------------------------------------------------------------------------------------------------------------------

		## First Plot (NCC)
		a[1,0].errorbar(x_values, y_values_mean_ncc, yerr = y_values_std_ncc,  fmt='-o', Label="Duplicated blots (with rotation)", color='orange', ecolor='orange')
		a[1,0].set_xlabel("Rotation Angle (degrees)")
		a[1,0].set_ylabel("NCC Coefficient")
		a[1,0].set_ylim(0, 1.3)
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_ncc,control_ncc), "r-", label = "Non-duplicated blots (without rotation)")
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_min_ncc, control_min_ncc), "r--")
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_max_ncc, control_max_ncc), "r--")
		a[1,0].legend(loc="upper left", prop={'size':9})

		## Second Plot (NCC)
		a[1,1].bar(x_values, y_values2_mean_ncc, 1, yerr = y_values2_std_ncc, color='orange')
		a[1,1].set_ylabel("Similarity")
		a[1,1].set_xlabel("Rotation Angle (degrees)")
		a[1,1].set_ylim(0,1.15)
		a[1,1].plot((np.min(x_values),np.max(x_values)), (1,1), "r--", label="Zero Distance")
		a[1,1].legend(loc="upper left", prop={'size':9}) 

		## --------------------------------------------------------------------------------------------------------------------------------------------------



	def scalingPerformance(self, figsize, title, show_example=False):
		"""
		plots performance of each descriptor method as a function of scaling factor. Performance is measured by L2 euclidean distance between orignal 
		blot and copied blit of given scaling factor.

		Parameters:
		------------
		figsize: tuple (x-dimension, y-dimension)
			sets the figure size
		title: str
			sets title of figure
		show_eaxample: boolean (default: False)
			If True, plots an example of one blot at each scale
			
		"""
		## Plot example figure of rotated blots

		if show_example is True:
			exp_blots = self.scaleDicts[0]
			exp_fig, exp_ax = plt.subplots(1,len(exp_blots["image"]), figsize=(10,2))

			for i, a in enumerate(exp_ax):
				a.imshow(exp_blots["image"][i], "gray")
				a.set_title(str(exp_blots["scaleFactor"][i]))
				a.axis("off")

		## compute euclidean distance as a function of scale ratio

		x_values = [] # intialize x-values array for plot
		y_values_des = [] # intitialize y-values array for plot
		y_values_ncc = []
		control_dist_des = [] # intialize control distribution to compare against variantion across images
		control_dist_ncc = []

		for s_dict in self.scaleDicts: #self.scaleDicts[0:8] + self.scaleDicts[9:12] + self.scaleDicts[13:]: # loop through each image-scale dictionary
			scaleFactor = s_dict["scaleFactor"]
			blots_scale = s_dict["image"]
			image_og = s_dict["original"]

			## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
			WB_og = wb.WesternBlot()
			WB_og.figure = image_og
			WB_og.figure_gray = cv2.cvtColor(image_og, cv2.COLOR_BGR2GRAY)
			WB_og.westernBlotExtractor()
			WB_og.blotExtractorWatershed(255, Visualize=False)

			des_og = WB_og.fingerprint["descriptor"][0]
			ncc_og = WB_og.fingerprint["blots"][0]

			distances_des = []
			distances_ncc = []
			for blot in blots_scale:
				## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
				WB = wb.WesternBlot()
				WB.figure = blot
				WB.figure_gray = cv2.cvtColor(blot, cv2.COLOR_BGR2GRAY)
				WB.westernBlotExtractor()
				WB.blotExtractorWatershed(255, Visualize=False)
	
				des = WB.fingerprint["descriptor"][0]
				ncc = WB.fingerprint["blots"][0]
				
				distances_des.append(np.linalg.norm(des - des_og)) # compute L2-distance between original image and rotated form
				if ncc_og.size >= ncc.size or ncc_og.size < ncc.size:
					ncc_og_scaled = cv2.resize(ncc_og, (ncc.shape[1], ncc.shape[0]), interpolation=cv2.INTER_NEAREST)
					distances_ncc.append(np.max(cv2.matchTemplate(ncc_og_scaled, ncc, cv2.TM_CCOEFF_NORMED)))


			x_values.append(scaleFactor) # append each corresponding angle row to x-value array
			y_values_des.append(distances_des) # append each rotated image row to y-value array
			y_values_ncc.append(distances_ncc)
			control_dist_des.append(des_og) # append each original image descriptor for the control line
			control_dist_ncc.append(ncc_og)

		## common x_values for each plot
		x_values = np.mean(np.array(x_values), axis=0) # corresponding angles 

		## ------------------------------------------------------------------------------------------------

		## solve for the control-value line
		# TODO Put in error if len(self.control_dist) is less than or equal to one
		control_dist_des = list(itertools.combinations(control_dist_des, 2))
		control_des = np.mean([np.linalg.norm(np.diff(c)) for c in control_dist_des])
		control_max_des= np.max([np.linalg.norm(np.diff(c)) for c in control_dist_des])
		control_min_des= np.min([np.linalg.norm(np.diff(c)) for c in control_dist_des])

		## first plot results (HOG)
		y_values_mean_des = np.mean(np.array(y_values_des), axis=0) # average over images
		y_values_std_des = np.std(np.array(y_values_des), axis=0) / np.sqrt(np.array(y_values_des).shape[0])
		
		## second plot results (HOG)
		y_values2_mean_des = (control_des - y_values_mean_des) / control_des
		y_values2_std_des = (control_des - (y_values_mean_des-y_values_std_des)) / control_des
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
		control_max_ncc = np.max(control_ncc)
		control_min_ncc = np.min(control_ncc)
		control_ncc = np.mean(control_ncc)

		## first plot resilts (NCC)
		y_values_mean_ncc = np.mean(np.array(y_values_ncc), axis=0)
		y_values_std_ncc = np.std(np.array(y_values_ncc), axis=0) / np.sqrt(np.array(y_values_ncc).shape[0])

		## second plot results (NCC)
		y_values2_mean_ncc = -(control_ncc - y_values_mean_ncc) / (1 - control_ncc + np.finfo(float).eps)
		y_values2_std_ncc = np.abs(control_ncc - np.abs(y_values_mean_ncc - y_values_std_ncc)) / (1 - control_ncc + np.finfo(float).eps)
		y_values2_std_ncc = np.abs(y_values2_mean_ncc - y_values2_std_ncc)

		## ------------------------------------------------------------------------------------------------


		## plot results
		f, a = plt.subplots(2,2,figsize = figsize)

		f.suptitle(title + " (N = "+str(len(y_values_des))+")", fontsize=18)

		## -------------------------------------------------------------------------------------------------------------------------------

		## First Plot (HOG)
		a[0,0].errorbar(x_values, y_values_mean_des, yerr = y_values_std_des, fmt='-o', label="Duplicated blots (with scaling)")
		a[0,0].set_xlabel("Scale")
		a[0,0].set_ylabel("Mean Euclidean Distance")
		a[0,0].set_ylim(0, control_des+.65)
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_des,control_des), "r-", label = "Non-duplicated blots (without scaling)")
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_min_des, control_min_des), "r--")
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_max_des, control_max_des), "r--")
		a[0,0].legend(loc="upper left", prop={'size':9})

		## Second Plot (HOG)
		a[0,1].bar(x_values, y_values2_mean_des, .05, yerr = y_values2_std_des)
		a[0,1].set_xlabel("Scale")
		a[0,1].set_ylabel("Similarity")
		a[0,1].set_ylim(0,1.15)
		a[0,1].plot((np.min(x_values),np.max(x_values)), (1,1), "r--", label="Zero Distance")
		a[0,1].legend(loc="upper left", prop={'size':9})  


		## --------------------------------------------------------------------------------------------------------------------------------

		## First Plot (NCC)
		a[1,0].errorbar(x_values, y_values_mean_ncc, yerr = y_values_std_ncc,  fmt='-o', Label="Duplicated blots (with scaling)", color='orange', ecolor='orange')
		a[1,0].set_xlabel("Scale")
		a[1,0].set_ylabel("NCC Coefficient")
		a[1,0].set_ylim(0, 1.3)
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_ncc,control_ncc), "r-", label = "Non-duplicated blots (without scaling)")
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_min_ncc, control_min_ncc), "r--")
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_max_ncc, control_max_ncc), "r--")
		a[1,0].legend(loc="upper left", prop={'size':9})

		## Second Plot (NCC)
		a[1,1].bar(x_values, y_values2_mean_ncc, .05, yerr = y_values2_std_ncc, color='orange')
		a[1,1].set_ylabel("Similarity")
		a[1,1].set_xlabel("Scale")
		a[1,1].set_ylim(-.15, 1.2)
		a[1,1].plot((np.min(x_values),np.max(x_values)), (1,1), "r--", label="Zero Distance")
		a[1,1].legend(loc="upper left", prop={'size':9}) 

		## ----------------------------------------------------------------------------------------------------------------------------------



	def adjContrastPerformance(self, figsize, title, show_example=False):
		"""
		plots performance of each descriptor method as a function of contrast factor. Performance is measured by L2 euclidean distance between original 
		blot and copied blot of given scaling factor.

		Parameters:
		------------
		figsize: tuple (x-dimension, y-dimension)
			sets the figure size
		title: str
			sets title of figure
		show_eaxample: boolean (default: False)
			If True, plots an example of one blot at each contrast	
		"""
		## Plot example figure of rotated blots

		if show_example is True:
			exp_blots = self.adjConDicts[0]
			exp_fig, exp_ax = plt.subplots(2,len(exp_blots["image"]), figsize=(10,3))

			for i, a in enumerate(exp_ax[0]):
				a.imshow(exp_blots["image"][i], "gray")
				a.set_title(str(exp_blots["contrastFactor"][i]))
				a.axis("off")
			for i, a in enumerate(exp_ax[1]):
				a.hist(cv2.cvtColor(exp_blots["image"][i], cv2.COLOR_BGR2GRAY).flatten(), 50, [0,256])
				a.set_xlim(0,255)
				a.axis("off")

		## compute euclidean distance as a function of scale ratio

		x_values = [] # intialize x-values array for plot
		y_values_des = [] # intitialize y-values array for plot
		y_values_ncc = []
		control_dist_des = [] # intialize control distribution to compare against variantion across images
		control_dist_ncc = []

		for c_dict in self.adjConDicts: # loop through each image-scale dictionary
			contrastFactor = c_dict["contrastFactor"]
			blots_contrast = c_dict["image"]
			image_og = c_dict["original"]

			## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
			WB_og = wb.WesternBlot()
			WB_og.figure = image_og
			WB_og.figure_gray = cv2.cvtColor(image_og, cv2.COLOR_BGR2GRAY)
			WB_og.westernBlotExtractor()
			WB_og.blotExtractorWatershed(255, Visualize=False)

			des_og = WB_og.fingerprint["descriptor"][0]
			ncc_og = WB_og.fingerprint["blots"][0]

			distances_des = []
			distances_ncc = []
			for blot in blots_contrast:
				## Compute interest-point descriptors using watershed blot extractor and HOG descripter calculation for original image
				WB = wb.WesternBlot()
				WB.figure = blot
				WB.figure_gray = cv2.cvtColor(blot, cv2.COLOR_BGR2GRAY)
				WB.westernBlotExtractor()
				WB.blotExtractorWatershed(255, Visualize=False)
	
				des = WB.fingerprint["descriptor"][0]
				ncc = WB.fingerprint["blots"][0]
				
				distances_des.append(np.linalg.norm(des - des_og)) # compute L2-distance between original image and rotated form
				if ncc_og.size >= ncc.size or ncc_og.size < ncc.size:
					ncc_og_scaled = cv2.resize(ncc_og, (ncc.shape[1], ncc.shape[0]), interpolation=cv2.INTER_LINEAR)
					distances_ncc.append(np.max(cv2.matchTemplate(ncc_og_scaled, ncc, cv2.TM_CCOEFF_NORMED)))
				# else:
				# 	ncc_scaled = cv2.resize(ncc, (ncc_og.shape[1], ncc.shape[0]), interpolation=cv2.INTER_NEAREST)
				# 	distances_ncc.append(np.max(cv2.matchTemplate(ncc_og, ncc_scaled, cv2.TM_CCOEFF_NORMED)))

			x_values.append(contrastFactor) # append each corresponding angle row to x-value array
			y_values_des.append(distances_des) # append each rotated image row to y-value array
			y_values_ncc.append(distances_ncc)
			control_dist_des.append(des_og) # append each original image descriptor for the control line
			control_dist_ncc.append(ncc_og)


		## common x_values for each plot
		x_values = np.mean(np.array(x_values), axis=0) # corresponding angles 

		## ------------------------------------------------------------------------------------------------

		## solve for the control-value line
		# TODO Put in error if len(self.control_dist) is less than or equal to one
		control_dist_des = list(itertools.combinations(control_dist_des, 2))
		control_des = np.mean([np.linalg.norm(np.diff(c)) for c in control_dist_des])
		control_max_des= np.max([np.linalg.norm(np.diff(c)) for c in control_dist_des])
		control_min_des= np.min([np.linalg.norm(np.diff(c)) for c in control_dist_des])

		## first plot results (HOG)
		y_values_mean_des = np.mean(np.array(y_values_des), axis=0) # average over images
		y_values_std_des = np.std(np.array(y_values_des), axis=0) / np.sqrt(np.array(y_values_des).shape[0])
		
		## second plot results (HOG)
		y_values2_mean_des = (control_des - y_values_mean_des) / control_des
		y_values2_std_des = (control_des - (y_values_mean_des-y_values_std_des)) / control_des
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
		control_max_ncc = np.max(control_ncc)
		control_min_ncc = np.min(control_ncc)
		control_ncc = np.mean(control_ncc)

		## first plot resilts (NCC)
		y_values_mean_ncc = np.mean(np.array(y_values_ncc), axis=0)
		y_values_std_ncc = np.std(np.array(y_values_ncc), axis=0) / np.sqrt(np.array(y_values_ncc).shape[0])

		## second plot results (NCC)
		y_values2_mean_ncc = -(control_ncc - y_values_mean_ncc) / (1 - control_ncc + np.finfo(float).eps)
		y_values2_std_ncc = np.abs(control_ncc - np.abs(y_values_mean_ncc - y_values_std_ncc)) / (1 - control_ncc + np.finfo(float).eps)
		y_values2_std_ncc = y_values2_mean_ncc - y_values2_std_ncc

		## ------------------------------------------------------------------------------------------------


		## plot results
		f, a = plt.subplots(2,2,figsize = figsize)

		f.suptitle(title + " (N = "+str(len(y_values_des))+")", fontsize=18)

		## -------------------------------------------------------------------------------------------------------------------------------

		## First Plot (HOG)
		a[0,0].errorbar(x_values, y_values_mean_des, yerr = y_values_std_des, fmt='-o', label="Duplicated blots (with cont. adj.)")
		a[0,0].set_xlabel("Contrast")
		a[0,0].set_ylabel("Mean Euclidean Distance")
		a[0,0].set_ylim(0, control_des+.65)
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_des,control_des), "r-", label = "Non-duplicated blots (without cont. adj.)")
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_min_des, control_min_des), "r--")
		a[0,0].plot((np.min(x_values),np.max(x_values)), (control_max_des, control_max_des), "r--")
		a[0,0].legend(loc="upper left", prop={'size':9})

		## Second Plot (HOG)
		a[0,1].bar(x_values, y_values2_mean_des, .05, yerr = y_values2_std_des)
		a[0,1].set_xlabel("Contrast")
		a[0,1].set_ylabel("Similarity")
		a[0,1].set_ylim(0,1.15)
		a[0,1].plot((np.min(x_values),np.max(x_values)), (1,1), "r--", label="Zero Distance")
		a[0,1].legend(loc="upper left", prop={'size':9})  


		## --------------------------------------------------------------------------------------------------------------------------------

		## First Plot (NCC)
		a[1,0].errorbar(x_values, y_values_mean_ncc, yerr = y_values_std_ncc,  fmt='-o', Label="Duplicated blots (with cont. adj.)", color='orange', ecolor='orange')
		a[1,0].set_xlabel("Contrast")
		a[1,0].set_ylabel("NCC Coefficient")
		a[1,0].set_ylim(0, 1.3)
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_ncc,control_ncc), "r-", label = "Non-duplicated blots (without scaling)")
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_min_ncc, control_min_ncc), "r--")
		a[1,0].plot((np.min(x_values),np.max(x_values)), (control_max_ncc, control_max_ncc), "r--")
		a[1,0].legend(loc="upper left", prop={'size':9})

		## Second Plot (NCC)
		a[1,1].bar(x_values, y_values2_mean_ncc, .05, yerr = y_values2_std_ncc, color='orange')
		a[1,1].set_ylabel("Similarity")
		a[1,1].set_xlabel("Contrast")
		a[1,1].set_ylim(0,1.15)
		a[1,1].plot((np.min(x_values),np.max(x_values)), (1,1), "r--", label="Zero Distance")
		a[1,1].legend(loc="upper left", prop={'size':9}) 

		## ----------------------------------------------------------------------------------------------------------------------------------




		

