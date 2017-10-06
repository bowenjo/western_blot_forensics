import numpy as np 
import cv2
import matplotlib.pyplot as plt
import wbFingerprintUtils as utils
from westernBlotClassifier import SVM_wb_images
from westernBlotClassifier import SVM_blots 
import json
from scipy import ndimage as ndi 
from scipy.stats import multivariate_normal
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import time



"""
CLASS FOR FINGERPRINTING WESTERN BLOT PLOTS FOR IMAGE FORENSICS
"""
##TODO: 
	  #	X for blots with connected portions: Look into a more refined object recognition algorithm (watershed segmentation).
	  # X refine area selection/filtering in utils.
	  # X add in work around to None blots if they occur.
	  # X design a better equalization function that ignores black background(utils). 
	  # X decide which angle measurement is better: from moments or from fitted ellipse
	  # X correct so will work if blots are originally white (inverse images) <--did not change for blot extractor original
	  # X Create Visualize instances for blotextractor function to clean up code
	  # X Create append functions for westerblotextractor function (start line 207)
	  # X remove old watershed function and put it in utils to clean up blotExtractorWatershed
	  # X Add in more global variables where needed
	  # X add in general filtering for blotextractorwatershed --> Filter out impossible shapes (i.e very small width and/or length for blots)
	  # X Work on the oversegmentation problem of the gradient-weighted watershed transform
	  #  - gradient-weighted distance transform seems to solve undersegmentation problem (mostly), but leads to oversegmentation. 
	  # Work on filtering the images more in utils.approxAndExtractRect to get rid of false wetern blot images.
	  # And/Or work on a blot-shape or blot-intensity stats to create a confidence score to distinguish non-blots from blots
	  # Delete false blots that seem to catch the corners of images due to image outlines and imperfection in extracting rectangles.  
	  # clean up utils.LocalMinimaDetector
	  # Update all paramaters/return comments and check code for comments/add comments where necessary.
	  # Look through code and make notes to increase readability and report below
	  #		-
	  #		-
	  #		- 
	  # Add in global variables for key parameters
	  # fix imshowblots and plothistogramandcdf functions
	  # fix betweenblots descriptor function
	  # Think about breaking up utils into seperate python files for cleaner readability

"""
GLOBAL PARAMETERS 
""" 
## -------LOADING-INFORMATION---------
LOAD_PATH_TO_FILE = "testFigures/nature_wb_output/output_revised.json" # path to .json dictionary from scrapy webscraping famework results
LOAD_FOLDER_NAME = "nature_wb_output" # folder name with scraped images

## -------OUTPUT-INFORMATION----------
OUTPUT_PATH_TO_FINGERPRINT = "testFigures/nature_wb_output/fingerprintDatabase_testNewSegMeth.npy" # output file path for fingerprint dictionary

## -------FIGURE-PROCESSING-----------
FIG_CNT_AREA_UPPR = 1000 # Subtracted from figure size to give upper area limit for computed contours 
FIG_CNT_AREA_LWR = .05 # percentage of figure size to give lower area limit for computed contours
FIG_CNT_ARC_LEN_APPROX = .18 # percentage of arc length of contours to approx cnt shape
FIG_IMG_SIZE_LIM = (2.8,1) # (upper,lower) number of std dev away from th mean of image sizes in figure
FIG_IMG_AR_LIM = (2,1) # (upper,lower) number of std dev away from the mean of image aspect ratios in figure
FIG_IMG_MODE_LIM = (3,1) # (upper,lower) number of std dev away from the mean of intensity modes in figure 
FIG_IMG_HIST_SHIFT_LEN = 4 # number of grayscale levels away from mode to accumulate for HIST_LIM
FIG_IMG_HIST_LIM = 2.0 # limit oformode/sum(hist(SHIFT_LEN))

## -------IMAGE-PREPROCESSING---------
PRE_MED_FILT = 5 # median filter kernel size for local min border drawing algorithm
PRE_GAUSS_FILT = 9 # gaussian filter kerenl size for local min border drawing algorithm
PRE_LOC_MIN_THRESH = 65 # gray-scale value threshold for local min (local min must have greater intensity than this value)
PRE_GLOBAL_MIN_NBRH = 2 # square neighborhood width for local min (local min must be global min within this square region)
PRE_DIST_UPPRBND = 15 # L2-distance from each local min (localmin must not be closer than this L2 distance)
PRE_SLOPE_ANGL = (75,2) # (upper, lower) slope angle for fitted line in border drawing algorith (fit lines within this range are deleted)

## -------WATERSHED-ALGORITHM---------
WTRSHD_MED_FILT = 5 # median filter kernel size for watershed algorithm
WTRSHD_STRUC_ELEM = 7 # structuring element kernel size for watershed algorithm
WTRSHD_GAUSS_FILT = 5 # gaussian filter kernel size for watershed algorithm
WTRSHD_GRAD_WEIGHT = 2 # weight that the morphological gradient has over distance transform

## -------IMAGE-POSTPROCESSING--------
POST_MERGE_WIDTH_LIM = .85 # percentage of maximum blot width to execute merging on
POST_MERGE_DIST_W_LIM = 3 # fraction of combined half-width of paired contours
POST_MERGE_DIST_H_LIM = 2 # fraction of comined half-height of paired contours
POST_MERGE_CS_RATIO = 1.2 # pre to post-merge confidence score ratio limit
POST_MAX_DES_PER_IMG = 40 # maximum number of blot descriptors per image (any more than this indicate false western blot images)
#MU = np.load("testFigures/nature_wb_output/GaussianModelParams.npy").item()["mean"]
#COV = np.load("testFigures/nature_wb_output/GaussianModelParams.npy").item()["covariance"]

## -------HOG-DESCRIPTOR--------------
HOG_BLOCK_SHAPE = (6,24) # block shape to resample extracted blot key point
HOG_BIN_SHAPE = (2,4) # bin shape to smaple orientations in
HOG_ORIENTATIONS = 8 # number of evenly spaced orientations to sample between 0 to 360 degrees.



THRESH_BLOCK_SIZE = 37
THRESH_C = 10
MASK_BORDER_SIZE = 7

SE_K_SIZE = 3
T_PARAM = 1
ORDER_PARAM = (0.25, 0.015)
VOTE_PARAM= (0.1, 0.1)
DT_PARAM = (.15, .025) # (0.15, 0.075) 
LINE_PARAM = -2 # -1
CONF_SCORE_PARAM = 1
DT_WEIGHT = 1
MV_DIST_PARAM = np.load("C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/testFigures/nature_wb_output/wbImageStatsParams.npy").item() 

def loadWesternBlotData(filename):
	"""
	loads in western blot .json file from scraped html. 

	Parameters:
	------------
	filename: str
		path to .json file of scraped images

	Returns:
	------------
	data: list
		list of dictionaries with keys:
			"images": contains information of path to image on disc
			"image_urls": western blot image url 
			"location_info": could be article title, url, authors, publisher, etc
	"""
	dir_dictionary = json.loads(open(filename).read())
	#print("There Are %r images in this set"%(len(dir_dictionary)))
	return(dir_dictionary)



class WesternBlot(object):
	""" 
	Class that encapsulates fingerprinting western blot images for digital forensics. To be performed on single, isolated western blot images. 
	Creates a dictionary containing pixel information and statistical properties for each blot in the image. Elements in the dictionary
	are split into units where each unit is a representation of a single blot. Blot images can be normalized for scale, rotation, and contrast.
	Translational invariance comes from extracting each blot seperately as a single template unit. The class also includes plotting functions
	that can be called after object.blotExtractor() has been executed. 
	"""

	def __init__(self, dir_dictionary=None, label=None, folder=None):

		# read in the figure from folder
		if dir_dictionary is not None:
			self.figure = self.readFileFromFolder(dir_dictionary, label = label, folder = folder) # self.figure = cv2.imread(dir_dictionary)
			# convert to grayscale if not None
			if self.figure is not None:
				self.figure_gray = cv2.cvtColor(self.figure, cv2.COLOR_BGR2GRAY)
			try:
				self.tags = {"image_url": self.readFileFromFolder(dir_dictionary, label='image_urls', folder=None),
							 "article_title": self.readFileFromFolder(dir_dictionary, label='title', folder=None)
							}
			except KeyError:
				raise KeyError("Check to make sure figure tags are correct")
		# if no dir_dictionary is given
		else:
			self.figure = None
			self.figure_gray = None
			self.tags = None
			#print("No figures were retrieved. Please define WesternBlot.figure, WesternBlot.figure_gray, and WesternBlot.tags to continue")

		# initialize final fingerprint attribute
		self.images = None
		self.Fingerprint = {"figure_tags": [], # tags regarding the article title and figure url
							"image_locs": [], # image location in figure 
							"blot_locs": [], # blot location in image
							"feature_vectors": [], # feature vector (HOG descriptor)
					   		}


		# Global variables
		self.ELLIPSE_CORR = 4 # corrects noise due to ellipse overfitting
		# Global filter variables
		self.FILTER_CNT_PTS = 5 # ellipse fitting requires at least 5 contour points
		self.FILTER_BLOT_SIZE_MIN = 0.005 # set to a fraction of image_gray.size to filter too small objects
		self.FILTER_BLOT_SIZE_MAX = 0.90 # set to a fraction of image_gray.size to filter too large objects
		self.FILTER_BLOT_DES_PARAMS = np.load("ideal_descriptor.npy") # load to filter out none "blot-like" descriptors
		self.FILTER_L2_LIM = .8 # L2 distance limit for "blot-like" descriptors
		self.FILTER_AR_MIN = 1/5
		self.FILTER_AR_MAX = 2

	def readFileFromFolder(self, dir_dictionary, label, folder=None):
		"""
		Picks out file from folder as directed by direction dictionary and label. Works with .json file formats used to store 
		.jpeg image url dictionaries from scrapy, the web-scrapping framework.

		Parameters:
		-------------
		dir_dictionary: python dictionary
			direction dictionary housing path to files
		label: str
			direction string to file type in folder:
		folder: str (default: None)
			path to folder name housing output files.

		Returns:
		-------------
		files: .jpeg image
			if label == 'images': outputs .jpeg images in folder
			else: outputs dir_dictionary["label"]

		"""
		if label == 'images': # 'images' is the main file label where the .jpeg files are located
			if folder is None:
				try:
					folder = str(input("Input folder name: ")) # if forgot to specify folder type, chance to input it
				except ValueError: 
					print("That was not a valid folder name. Please try again") # if input is not a string, raise error

			try:
				file = cv2.imread("testFigures/" + folder + '/{}'.format(dir_dictionary[label][0]["path"])) # load in image if label is 'image' 
			except IndexError:
				return(None)
		else:
			file = dir_dictionary[label] # otherwise return the dictionary element requested by key

		if file is not None:
			return(file) # returns file if file exists
		else:
			pass # otherwise, continue searching for other files

	def westernBlotExtractor(self, break_point=0.90, maxValue=255, threshold=75, VISUALIZE=False):
		""" 
		Extracts western blot images from scientific figures

		Parameters:
		-------------
		clf: python class object
			pre-trained support vector machine object used to differenciate between wetern blot and not western blot
		break_point: float (default = 0.90)
			defines the contrast threshold for ectified function
		maxValue: uint8 (default = 255)
			The maximum value in thresholding function
		threshold: uint8 (default = 75)
			defines the thresholding value in thresholding function
		VISUALIZE: boolean (default: False)
			if true, outputs ploting functions to show image segmentation results
		WATERSHED: str (default: "gw-DT")
			watershed method

		Returns:
		-------------
		None
		"""

		cFiltered = utils.contrastFilters(self.figure_gray, break_point = break_point) # apply a filter to exagerate contrast between mid-level grayvalues and white  
		thresh = cv2.threshold(cFiltered, threshold, maxValue, cv2.THRESH_BINARY_INV)[1] # thresholds image to make contour detection easier
		contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] # find contours in the figure

		# filter contours by size (i.e. discard contours too small or too large)
		contours = utils.filterContourArea(contours, np.size(self.figure_gray), alpha=FIG_CNT_AREA_UPPR, beta=FIG_CNT_AREA_LWR) 

		# approximate each contour to extract only rectangles
		self.images = utils.approxAndExtractRect(self.figure, contours, 
											epsilon=FIG_CNT_ARC_LEN_APPROX, 
											size_lim=FIG_IMG_SIZE_LIM, 
											whr_lim=FIG_IMG_AR_LIM,
											mode_lim=FIG_IMG_MODE_LIM,
											shift_length=FIG_IMG_HIST_SHIFT_LEN, 
											hist_lim=FIG_IMG_HIST_LIM,
											)
		if self.images is not None:
			# visulaize results
			if VISUALIZE is True:
				self.visualizeWbImages(self.figure_gray, thresh, contours, self.images["grays"], figsize=(6,6))

			# loop over each image in figures:
			none_check = 0
			for _, image in enumerate(self.images["images"]):
				# call watershed method to collect blot descriptors and locations
				fingerprint, locs = self.blotExtractor(image, maxValue=255, inv_thresh=100, se_k = SE_K_SIZE, THRESH_TYPE='adaptive', VISUALIZE=False)
				if fingerprint is None or locs is None:
					continue
				# check if fingerprints were extracted or if too many were
				if len(fingerprint["blots"]) != 0 and len(fingerprint["blots"])<=POST_MAX_DES_PER_IMG:
					none_check += 1
					# append the blot data to self.Fingerprint dictionary
					self.appendBlotData(_, none_check, locs, fingerprint, self.images)

			# convert each to an array
			self.Fingerprint["image_locs"] = np.array(self.Fingerprint["image_locs"])

			# print error messages if lengths don't line up
			if len(self.Fingerprint["image_locs"]) != len(self.Fingerprint["feature_vectors"]):
				print("The lengths of the dictionary lists should be equal, something went wrong")
			if len(self.Fingerprint["figure_tags"]) != len(self.Fingerprint["feature_vectors"]):
				print("The lengths of the dictionary lists should be equal, something went wrong")

		else:
			self.Fingerprint = None

	def visualizeWbImages(self, figure, thresh, contours, images, figsize):
		"""
		visualize the western blot figure segmentation method. Plots original figure 
		"""
		copy = figure.copy()
		cv2.drawContours(copy, contours, -1, (0,0,255), 2)
		plt.figure(figsize=figsize), plt.imshow(figure, "gray"), plt.title("Original figure with image contours") 
		plt.figure(figsize=figsize), plt.imshow(thresh, "gray"), plt.title("Thresholded figure with contrast adjustment")
		for i, image in enumerate(images):
			f,a = plt.subplots(1,2) # initialize subplots
			a[0].imshow(image, "gray"), a[0].set_title("Extracted image %r"%(i))
			a[1].hist(image.flatten(), 256, [0,256], color = 'r')
			a[1].set_title("Mean: %r & Variance: %r"% (round(np.mean(image.flatten()), 2), round(np.var(image.flatten()), 2)))
			a[1].set_xlim([0,255])
		plt.show()

	def appendBlotData(self, index, none_check, locs, fingerprint, images):
		"""
		appends blot data to Fingerprint dictionary with given index
		"""
		if none_check == 1:
			locs, fingerprint = self.reshapeBlotData(locs, fingerprint)

			self.Fingerprint["figure_tags"] = [self.tags]*len(fingerprint["descriptor"])
			self.Fingerprint["image_locs"] = [images["loc"][index]]*len(fingerprint["descriptor"])
			self.Fingerprint["blot_locs"] = locs["loc"]
			self.Fingerprint["feature_vectors"] = fingerprint["descriptor"]
			#self.Fingerprint["feature_vectors"] = fingerprint["shape_stats"]
		else:
			locs, fingerprint = self.reshapeBlotData(locs, fingerprint)
			
			self.Fingerprint["figure_tags"] += [self.tags]*fingerprint["descriptor"].shape[0]
			self.Fingerprint["image_locs"] += [images["loc"][index]]*fingerprint["descriptor"].shape[0] 
			self.Fingerprint["blot_locs"] = np.vstack((self.Fingerprint["blot_locs"], locs["loc"]))
			self.Fingerprint["feature_vectors"] = np.vstack((self.Fingerprint["feature_vectors"], fingerprint["descriptor"]))
			#self.Fingerprint["feature_vectors"] = np.vstack((self.Fingerprint["feature_vectors"], fingerprint["shape_stats"]))

	def reshapeBlotData(self, locs, fingerprint):
		"""
		reshapes blot data if 0-axis (row-axis) is None
		"""
		if locs["loc"].shape[0] is None:
			locs["loc"] = locs["loc"].reshape(1, np.size(locs["loc"]))
			print("Reshaped!")
		if fingerprint["descriptor"].shape[0] is None:
			fingerprint["descriptor"] = fingerprint["descriptor"].reshape(1, np.size(fingerprint["descriptor"]))
			print("Reshaped!") 

		return(locs, fingerprint)

	def blotExtractor(self, image, maxValue, inv_thresh, se_k, THRESH_TYPE, VISUALIZE=False, EQUALIZE=False):
		"""
		Parameters:
		---------------
		image: numpy array
			western blot image to be segmented
		maxValue: uint8
			max gray-scale value for threshold
		inv_thresh: uint8
			if greater than this gray-scale value, image will be inverted
		se_k: int (odd)
			structuring element kernel size
		THRESH_TYPE: str
			'otsu' - uses otsu thresholding
			'adaptive' - uses an adaptive thresholding method
		VISUALIZE: bool (default: False)
			if True, outputs visualization figures

		Returns:
		---------------
		fingerprint: dic
			python dictionary containing the blot descriptor
		locs: dic
			python dictionary containing indexing and location information

		"""

		## Preprocessing:
		# ---------------------------------------------------------------------

		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray-scale
		border = self.createBorderMask(image_gray.shape, border_size=MASK_BORDER_SIZE)
		# if len(image_gray[np.where(border==0)]) > 0:
		# 	if np.mean(image_gray[np.where(border==0)]) < inv_thresh: 
		# 		image_gray = cv2.bitwise_not(image_gray) # invert thresh if westernblot image is inverted (i.e. dark background).

		# build structuring element
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_k,se_k)) 
		# Smooth and invert image
		image_gray_smooth = cv2.GaussianBlur(image_gray, (se_k, se_k), 0) # smooth image
		image_gray_inv = cv2.bitwise_not(image_gray_smooth) # foreground must be 'peaks' in image intensitiess

		# threshold the image
		# otsu
		thresh = cv2.threshold(image_gray_smooth, 0, maxValue, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # seperates foreground from background
		# adaptive
		thresh_ad_m = cv2.adaptiveThreshold(image_gray_smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, THRESH_BLOCK_SIZE, THRESH_C)

		# mophological operation
		if THRESH_TYPE == 'otsu':
			thresh = cv2.bitwise_and(thresh, thresh, mask=border)
			opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se)
		if THRESH_TYPE == 'adaptive':
			thresh_ad_m = cv2.bitwise_and(thresh_ad_m, thresh_ad_m, mask=border)
			closing = cv2.morphologyEx(thresh_ad_m, cv2.MORPH_CLOSE, se)
			opening = cv2.morphologyEx(thresh_ad_m, cv2.MORPH_OPEN, se)

		# build convex hull mask of objects in thresholded image
		mask = self.createConvexMask(opening) # maskes the the local minima to convex hull area
		if mask is None:
			return(None, None)

		## Segmenting Image From Local Minima:
		# -------------------------------------------------------------------------

		# define local minima parameters
		T = self.getT(image_gray_inv, alpha=T_PARAM, mask=opening) # set max intensity threshold for local min
		order = self.getShapePortion(image_gray.shape, alpha=ORDER_PARAM) # set order for local min detection
		dist_thresh = self.getShapePortion(image_gray.shape, alpha=DT_PARAM)
		vote_lim = None # TODO: make this adapt to image resolution

		# find local minima location given parameters
		h_coord, v_coord = utils.gridLocalMin(image_gray_inv, T, order) # find local minima coordinates
		#print(len(h_coord[0]), len(v_coord[0]))

		# find divide lines given local minima
		divide_line_image = utils.gridLineDetector(opening, h_coord, vote_lim=2, dist_thresh=dist_thresh[1],
													line_type = 'h', kernel_size=se_k, alpha = LINE_PARAM, mask=mask)
		divide_line_image = utils.gridLineDetector(divide_line_image, v_coord, vote_lim=15, dist_thresh=dist_thresh[0],
													line_type = 'v', kernel_size=se_k, alpha = LINE_PARAM, mask=mask)

		# TODO: combine 'h' and 'v' portions into one function

		## Performing Watershed Operation:
		# --------------------------------------------------------------------------

		# find markers
		# dtwImage_gray, dtwImage = utils.dtwGradient(image_gray_smooth, opening, DT_WEIGHT) # for performance plots only
		dtwImage_gray, dtwImage = utils.dtwGradient(image_gray_smooth, divide_line_image, DT_WEIGHT) 
		markers = utils.findMarkers(dtwImage_gray, divide_line_image) # starting points for watershed operation
		# define sure foreground, sure background and unkown region to be determined by watershed algorithm
		sure_fg = np.zeros_like(image_gray, dtype=np.uint8) 
		sure_fg[markers>0] = 255 # Sure foreground (i.e. we are sure this is an object)
		sure_bg = cv2.dilate(opening, se, iterations=2) # Sure background (i.e. we are sure this is not an object)
		unknown = cv2.subtract(sure_bg, sure_fg) # unknown region where watershed lines will be placed
		markers = markers + 1
		markers[unknown==255] = 0

		# watershed  
		labels = cv2.watershed(dtwImage, markers.copy())


		## Visualization:
		# ---------------------------------------------------------------------------

		if VISUALIZE:

			# surface plot visualization 

			# from mpl_toolkits.mplot3d import Axes3D
			# from matplotlib import cm
			# from matplotlib.ticker import LinearLocator, FormatStrFormatter


			# fig = plt.figure()
			# ax = fig.gca(projection='3d')

			# # Make data.
			# X = range(dtwImage_gray.shape[1])
			# Y = range(dtwImage_gray.shape[0])
			# X, Y = np.meshgrid(X, Y)
			# Z = cv2.bitwise_not(dtwImage_gray)
			# Z[border==0]=0

			# # Plot the surface.
			# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
			#                        linewidth=0, antialiased=False)

			# # Customize the z axis.
			# ax.set_zlim(0, 255)
			# ax.zaxis.set_major_locator(LinearLocator(10))
			# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

			# # Add a color bar which maps values to colors.
			# fig.colorbar(surf, shrink=0.5, aspect=5)

			# plt.show()


			local_min_image = image.copy()
			if h_coord is not None:
				local_min_image[h_coord] = (255,0,0)
			if v_coord is not None:
				local_min_image[v_coord] = (0,0,255)

			segmented_image = image.copy()
			segmented_image[labels==-1] = (255,0,0)

			# Preprocessing images
			f1, a1 = plt.subplots(2,2, figsize = (18,10))
			a1[0,0].imshow(image_gray, "gray"), a1[0,0].set_title("Original Image")
			a1[1,0].imshow(opening, "gray"), a1[1,0].set_title("Object Outline")
			a1[0,1].imshow(local_min_image), a1[0,1].set_title("Local Minima")
			a1[1,1].imshow(divide_line_image, "gray"), a1[1,1].set_title("Division Lines From Local Min")
			f1.tight_layout()

			# Watershed Segmentation
			f2, a2 = plt.subplots(2,2, figsize=(18,10))
			a2[0,0].imshow(dtwImage), a2[0,0].set_title("Distance Transform Weighted Image")
			a2[0,1].imshow(markers), a2[0,1].set_title("Markers for Wateshed")
			a2[1,0].imshow(labels), a2[1,0].set_title("Watershed Operation")
			a2[1,1].imshow(segmented_image), a2[1,1].set_title("Final Segmentation")
			f2.tight_layout()

		return(self.appendFingerprintData(image_gray, labels, start_idx=2, EQUALIZE=False))
		#return(self.mergeStep(image_gray, labels, start_idx=2, EQUALIZE=EQUALIZE))

	def getT(self, image, alpha, mask=None):
		"""
		Gets intensity threshold for local min detection as a function of mean and stdDev of masked image
		"""
		mean, stdDev = cv2.meanStdDev(image, mask=mask)
		return(mean - (alpha * stdDev))

	def getShapePortion(self, shape, alpha):
		"""
		Gets order for local min detection or number of votes for hough line transform as a function of image shape
		TODO: make more dynamic. have a cut-off for higher resolution images
		"""
		return(np.maximum(int(alpha[0]*shape[0]),1), np.maximum(int(alpha[1]*shape[1]),1))

	def createConvexMask(self, image):
		"""
		creates a convex mask (binary image) of foreground object of a binary input image 
		"""
		contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
		hull_image = np.zeros_like(image.copy(), dtype=np.uint8)
		if len(contours) == 0:
			return(None)
		for cnt in contours:
			hull = cv2.convexHull(cnt)
			cv2.drawContours(hull_image, [hull], 0, (255), -1)
		return(hull_image)

	def createBorderMask(self, image_shape, border_size):
		"""
		creates a mask to eliminate border imperfections
		"""
		borders = np.zeros(image_shape, dtype= np.uint8)
		pt1 = (border_size, border_size)
		pt2 = (image_shape[1] - border_size, image_shape[0] - border_size)
		cv2.rectangle(borders, pt1, pt2, (255), -1)

		return(borders)

	def mergeStep(self, image, markers, start_idx, EQUALIZE=False):
		# first pass through with desired watershed method
		fingerprint, locs = self.appendFingerprintData(image, markers, start_idx, EQUALIZE)
		# merge oversegemented blots as a function of width
		if len(fingerprint["blots"]) != 0:
			merged = utils.mergeWatershedLines(locs, image.shape,
											   alpha = POST_MERGE_WIDTH_LIM)
			# second pass through post merging
			fingerprint, locs = self.appendFingerprintData(image, merged, 1, EQUALIZE)
		else:
			return(fingerprint, locs)

		return(fingerprint,locs)

	def appendFingerprintData(self, image, markers, start_idx, EQUALIZE=False, clf=None):
		"""
		appends descriptor and location information to fingerprint and locs dictionaries
		"""
		# initialize feature-vector information
		fingerprint = {"image": image.copy(),
					   "blots": [],
					   "descriptor": [],
					  }

		# initial location information
		locs = {"centroids": [],
				"angles": [],
				"loc": [],
				"contours": []
			   }

		for mark in np.unique(markers)[start_idx:]:
			# Create a mask of the selected blot
			mask = np.zeros(image.shape, dtype=np.uint8)
			mask[markers==mark] = 255

			cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2] # find contour of the blot

 			# contours must not be an empty list
			if len(cnts) < 1:
				continue

			cnt_max = max(cnts, key = cv2.contourArea) # use maximum area contour

			# openCV requires the points in a contour to be above a certain threshold 
			if len(cnt_max) < self.FILTER_CNT_PTS:
				continue

			# check how well the new blot fits the blot-shape statistical model. 
			if self.checkBlotStatModel(cnt_max, image, 256):
				continue

			x,y,w,h = cv2.boundingRect(cnt_max) # extract blot region of interest for viewing purposes


			# fit ellipse to compute orientation angle and major and minor principal axes
			(cx, cy), (ma, MA), theta = cv2.fitEllipse(cnt_max) # fit ellipse
			theta = theta - 90 # opencv likes theta rotated relative to 90 degrees
			locs["centroids"].append((cx,cy)) # append centroids for further analysis
			locs["angles"].append(theta) # append angles for further analysis

			# normalize orientation of blot by rotating about centroid
			rotation_matrix = cv2.getRotationMatrix2D((cx, cy), theta, 1) # define the rotation matrix for angle theta
			rotated_blot = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0])) # rotate the image by angle theta
			block_y, block_x = utils.pickBlockSize(ma, MA-self.ELLIPSE_CORR, (4,4)) # make sure block shape is multiple of 4 for resampling
			rotated_blot = self.isolateRotatedBox(rotated_blot, cx, block_x, cy, block_y)

			# equalize image if true
			if EQUALIZE is True:
				rotated_blot = cv2.equalizeHist(rotated_blot)

			# extracted blot must be 2-dimensional
			if rotated_blot.shape[0] == 0 or rotated_blot.shape[1] == 0: 
				continue

			# compute blot descriptor: histogram of oriented gradients (magnitude and direction)
			des = utils.HOG(rotated_blot,
							blockShape = HOG_BLOCK_SHAPE, 
							binShape = HOG_BIN_SHAPE, 
							orientations = HOG_ORIENTATIONS, 
							L2_NORMALIZE = True)

			# descriptor must not be None object
			if des is None:
				continue

			# append blots and features to lists
			fingerprint["descriptor"].append(des)
			fingerprint["blots"].append(rotated_blot)
			locs["loc"].append((x,y,w,h))
			locs["contours"].append(cnt_max)

		# convert to numpy array to avoid compatability issues
		fingerprint["descriptor"] = np.array(fingerprint["descriptor"])
		locs["loc"] = np.array(locs["loc"])
		locs["contours"] = np.array(locs["contours"])
		
		return(fingerprint, locs)

	def isolateRotatedBox(self, rotated_blot, cx, block_x, cy, block_y ):
		"""
		isolates blot from rotated image and returns truncated array
		"""
		(cx, cy) = (int(round(cx)),int(round(cy))) # make sure int values

		# compute bounds for bounding box
		lbx = np.maximum(int(round(cx - (block_x/2))), 0) # lower and upper x-axis bounds for bounding rectangle
		ubx = np.maximum(int(round(cx + (block_x/2))), 0) 
		lby = np.maximum(int(round(cy - (block_y/2))), 0) # lower and upper y-axis bounds for bounding rectangle
		uby = np.maximum(int(round(cy + (block_y/2))), 0) 

		return(rotated_blot[lby:uby, lbx:ubx]) # extract rotated blot from image

	def checkBlotStatModel(self, contour, image, num_bins):
		blot_stats = utils.blot_stats(contour, image, num_bins) # record shape stats for new blot contour
		conf_score = multivariate_normal.pdf(blot_stats, mean=MV_DIST_PARAM["mean"], cov=MV_DIST_PARAM["covariance"]) # see how new contour fits the model
		#print(conf_score)

		# if "confidence score" (pdf) is less than user-chosen confidence score threshold, do not record 
		if conf_score < CONF_SCORE_PARAM or np.isnan(conf_score):
			return(True) # Skip: do not measure descriptor on this contour
		else:
			return(False) # Don't skip:  measure descriptor on this contour



	def betweenBlotAngles(self, epsilon, alpha):
		"""
		Computes a between-blot angle comparison. Produces a 2D weighted mean feature vector, weighted by euclidean distance between centers of blots.

		Parameters:
		------------
		epsilon: int
			determines the bounding size
		alpha: float
			determines the rate constant for the distnace weighting function

		Returns:
		------------ 
		"bb_desctiptor": list
			between blot angle comparison feature vector for each cardinal direction, weighted by euclidean distance between centroids.
		"""

		if len(self.centroids_angles["centroids"]) == 0:
			raise RuntimeError("Must execute blotExtractor before calling function")

		self.fingerprint["bb_descriptor"] = []

		centroids = np.array(self.centroids_angles["centroids"])
		angles = np.array(self.centroids_angles["angles"])

		for idx, centroid in enumerate(centroids):
			## build bounds
			x_bnd = (centroid[0] - epsilon, centroid[0] + epsilon)
			y_bnd = (centroid[1] - epsilon, centroid[1] + epsilon)

			## compile x-direction (column-wise) and y-direction (row-wise) centroids
			x_centers = np.array([centroids[i] for i, c in enumerate(centroids[:, 0]) if c >= x_bnd[0] and c <= x_bnd[1]])
			x_angles = np.array([angles[i] for i, c in enumerate(centroids[:, 0]) if c >= x_bnd[0] and c <= x_bnd[1]])
			y_centers = np.array([centroids[i] for i, c in enumerate(centroids[:, 1]) if c >= y_bnd[0] and c <= y_bnd[1]])
			y_angles = np.array([angles[i] for i, c in enumerate(centroids[:, 1]) if c >= y_bnd[0] and c <= y_bnd[1]])

			## calculate euclidian distance for each cardinal direction:
			x_dist = np.linalg.norm((x_centers - centroid), axis = 1)
			y_dist = np.linalg.norm((y_centers - centroid), axis = 1)

			## compute angular differences across each cardinal direction
			x_diff_angle = np.abs(x_angles - angles[idx])
			y_diff_angle = np.abs(y_angles - angles[idx])

			## compute weights
			wx = np.mean(x_diff_angle)  # * alpha * np.exp(-x_dist))
			wy = np.mean(y_diff_angle)  # * alpha * np.exp(-y_dist))

			## append weight pairs to between-blot feature vector
			self.fingerprint["bb_descriptor"].append([wx, wy])



	def imshowBlots(self, blots, title, draw_axes=False):
		"""
		plots an array of all blots in the image

		Parameters: 
		-------------
		title: str
			String that sets the title for the figure
		draw_axes: boolean (default: False)
			if True, draws principal axes and orientation for each blot

		"""

		num_blots = len(blots)
		subplot_shape = np.floor(np.sqrt(num_blots)) + 1

		f, axis = plt.subplots(int(subplot_shape),int(subplot_shape))
		f.suptitle(title)
		for i, ax in enumerate(axis.flatten()):
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

			if ax in axis.flatten()[0:num_blots]:
				blot = blots[i].copy()
				
				if draw_axes == True:
					rows, cols = blot.shape
					c = (int(cols/2), int(rows/2))
					theta = self.centroids_angles["angles"][i]
					(MA,ma) = self.fingerprint["axes"][i]

					## find max-axis point
					max_line_x = c[0] + (MA/2) * np.cos(theta*np.pi/180)
					max_line_y = c[1] - (MA/2) * np.sin(theta*np.pi/180)
					max_pt = (int(max_line_x), int(max_line_y))

					## find min-axis point
					min_line_x = c[0] - (ma/2) * np.cos(theta*np.pi/180-(np.pi/2))
					min_line_y = c[1] + (ma/2) * np.sin(theta*np.pi/180-(np.pi/2))
					min_pt = (int(min_line_x), int(min_line_y))

					cv2.line(blot, c, max_pt, (255), 1)
					cv2.line(blot, c, min_pt, (255), 1)

				ax.imshow(blot, "gray")


	def plotHistogramAndCdf(self, title, num_bins=256):
		"""
		plots histogram and cdf of all blots in an image

		Parameters: 
		-------------
		title: str
			String that sets the title for the figure
		num_bins: int (default: 256)
			defines the number of bins in histogram

		"""
		if len(self.fingerprint["blots"]) == 0:
			raise RuntimeError("Must execute blotExtractor before calling function")

		num_blots = len(self.fingerprint["blots"])
		subplot_shape = np.floor(np.sqrt(num_blots)) + 1

		f, axis = plt.subplots(int(subplot_shape),int(subplot_shape))
		f.suptitle(title)
		for i, ax in enumerate(axis.flatten()):
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			if ax in axis.flatten()[0:num_blots]:
				blot_wo_black = [x for x in self.fingerprint["blots"][i].flatten() if x!=0]
				# Plot histogram
				ax.hist(blot_wo_black, num_bins, [0,num_bins], color = 'r')
				ax.set_xlim([0,num_bins-1])
				# Plot cdf
				ax_clone = ax.twinx()
				hist, _ = np.histogram(blot_wo_black, np.arange(num_bins), density=False)
				cdf = hist.cumsum()
				ax_clone.plot(cdf, color = 'b')
				ax_clone.yaxis.set_visible(False)

def view(idx_list, figsize):
	# load in the western blot data
	data = loadWesternBlotData(LOAD_PATH_TO_FILE)

	# load in Fingerprints
	Fingerprints = np.load(OUTPUT_PATH_TO_FINGERPRINT).item()

	for i in idx_list:
		if i < 0 or i > Fingerprints["feature_vectors"].shape[0]:
			raise IndexError("an index in idx_list is out of bounds for Fingerprint with length: ", Fingerprints["feature_vecors"].shape[0])

		figure_idx = Fingerprints["figure_idx"][i]
		image_loc = Fingerprints["image_locs"][i]
		blot_loc = Fingerprints["blot_locs"][i]
		tags = Fingerprints["figure_tags"][i]

		figure = WesternBlot(data[figure_idx], label="images", folder="nature_wb_output").figure_gray
		image = figure[image_loc[1]:image_loc[1]+image_loc[3], image_loc[0]:image_loc[0]+image_loc[2]]
		blot = image[blot_loc[1]:blot_loc[1]+blot_loc[3], blot_loc[0]:blot_loc[0]+blot_loc[2]]

		figure = cv2.cvtColor(figure, cv2.COLOR_GRAY2BGR)
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

		cv2.rectangle(figure,
					 pt1 = (image_loc[0], image_loc[1]), 
					 pt2 = (image_loc[0]+image_loc[2], image_loc[1]+image_loc[3]),
					 thickness = 10,
					 color = (0,0,255))

		cv2.rectangle(image,
					  pt1 = (blot_loc[0], blot_loc[1]),
					  pt2 = (blot_loc[0]+blot_loc[2], blot_loc[1]+blot_loc[3]),
					  color = (0,0,255),
					  thickness = 2)

		f, ax = plt.subplots(1, 3, figsize = figsize)
		f.suptitle("Article title: " + str(tags["article_title"]))
		ax[0].imshow(figure, "gray"), ax[0].set_title("Figure"+str(figure_idx))
		ax[1].imshow(image, "gray"), ax[1].set_title("Image")
		ax[2].imshow(blot, "gray"), ax[2].set_title("Blot")
		f.tight_layout()

	plt.show()

def main(TO_FILE, FOLDER_NAME, TO_FINGERPRINT):
	## load in western blot data
	#data = loadWesternBlotData(TO_FILE)
	data = np.load(TO_FILE)

	none_check = 0
	time.clock()
	# loop over all directory dictionaries housing paths to western blot figures and metadata
	for _, dir_dictionary in enumerate(data):
		# create WesternBlot object 
		WB = WesternBlot(dir_dictionary, label="images", folder=FOLDER_NAME)
		if WB.figure is None:
			continue
		# call westernBlotExtractor attibute to create fingerprint dictinary with descriptors of extracted blots 
		WB.westernBlotExtractor(VISUALIZE=False)
		if WB.Fingerprint is not None:
			if np.size(WB.Fingerprint["feature_vectors"]) != 0:
				none_check+=1
				if none_check == 1:
					Fingerprints = WB.Fingerprint # intitialize Fingerprints dictionary with first available WB.Fingerprint
					Fingerprints["figure_idx"] = [_]*WB.Fingerprint["feature_vectors"].shape[0] # initialize indexing system to locate fingerprints
				else:
					# loop over keys in Fingerprint dictionary and append descriptors and metadata
					for key in WB.Fingerprint.keys():
						try:
							if key != "figure_tags": 
								# append via numpy's vstack to create stacked arrays of descriptors and metadata
								if WB.Fingerprint[key].shape[0] is None:
									WB.Fingerprint[key].reshape(1,np.size(WB.Fingerprint[key]))
								Fingerprints[key] = np.vstack((Fingerprints[key], WB.Fingerprint[key]))
							else:
								# concatenate lists if key is "figure_tags"
							 	Fingerprints[key] += WB.Fingerprint[key]
						except ValueError:
							print("Figure %r had the problem fingerprint and key: "%(_+1), len(WB.Fingerprint[key]), key)
							break
						except AttributeError:
							print("Figure %r had the problem fingerprint and key: "%(_+1), len(WB.Fingerprint[key]), key)
							break
					# append figure indexes 
					Fingerprints["figure_idx"] += [_]*WB.Fingerprint["feature_vectors"].shape[0]
			else:
				print("There were no feature vectors in the images of figure %s"%(_+1))

		else:
			print("There were no images in figure %s"%(_+1))

		# print progress report to command line every 10 figures
		if (_ + 1) % 10 == 0:
			print(round(time.clock()), "secs: Analyzed %r figures: %r total feature vectors extracted"%(_+1, Fingerprints["feature_vectors"].shape[0]))

	if Fingerprints["feature_vectors"].shape[0] != len(Fingerprints["figure_idx"]):
		print("The lengths of the dictionary lists should be equal, something went wrong")

	# save fingerprint in .npy format
	np.save(TO_FINGERPRINT, Fingerprints)

	# print final progress report to command line
	print(round(time.clock()), "secs: Completed. %r figures used and %r features extracted"% (len(np.unique(Fingerprints["figure_idx"])), Fingerprints["feature_vectors"].shape[0]))


if __name__ == '__main__':
	#main()
	#view([17200,17201], (9,5))

	## BMC_output_schedule:
	# batch 1 9/20
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch1(0,999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch1")
	# # batch 2 9/20
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch2(1000,1999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch2")
	# # batch 3 9/20
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch3(2000,2999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch3")
	# # batch 4 9/20
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch4(3000,3999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch4")
	# # batch 5 9/20
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch5(4000,4999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch5")
	# batch 6 9/21
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch6(5000,5999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch6")
	# # batch 7 9/21
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch7(6000,6999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch7")
	# # batch 8 9/21
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch8(7000,7999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch8")
	# batch 9 9/21
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch9(8000,8999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch9")
	# # batch 10 9/21
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch10(9000,9999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch10")
	# batch 11 9/21 error
	# main(TO_FILE = "testFigures/BMC_output/BMC_output_batch11(10000,10999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch11")
	# batch 12 9/24
	main(TO_FILE = "testFigures/BMC_output/BMC_output_batch12(11000,11999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch12")
	# batch 13 9/24
	main(TO_FILE = "testFigures/BMC_output/BMC_output_batch13(12000,12999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch13")
	# batch 14 9/24
	main(TO_FILE = "testFigures/BMC_output/BMC_output_batch14(13000,13999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch14")
	# batch 15 9/24
	main(TO_FILE = "testFigures/BMC_output/BMC_output_batch15(14000,14999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch15")
	# batch 16 9/24
	main(TO_FILE = "testFigures/BMC_output/BMC_output_batch16(15000,15999).npy" , FOLDER_NAME="BMC_output", TO_FINGERPRINT = "testFigures/BMC_output/BMC_output_fingerprint_batch16")
	
	print("Completed. Fingerprints saved in drive ")



