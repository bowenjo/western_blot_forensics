import numpy as np 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import time

import utils.wbFingerprintUtils as utils
import params.config as config
from classifiers.clf import wb_clf


"""
CLASS FOR GATHERING FINGERPRINTS ON WESTERN BLOT PLOTS FOR IMAGE FORENSICS
"""

class WesternBlot(object):
	""" 
	Class that encapsulates gathering fingerprints on western blot images for digital forensics. To be performed on 
	single, isolated western blot images. Creates a dictionary containing pixel information and statistical properties 
	for each blot in the image. Elements in the dictionaryare split into units where each unit is a representation of 
	a single blot. Blot images can be normalized for scale, rotation, and contrast.Translational invariance comes from
	extracting each blot seperately as a single template unit. The class also includes visualization functions
	"""

	def __init__(self, dir_dictionary=None, label=None, folder=None,
		 		 clf_image = None, clf_blot = None):

		self.clf_image = clf_image
		self.clf_blot = clf_blot
		# Get the figures from a directory if given
		if dir_dictionary is not None:
			self.figure = self.readFileFromFolder(dir_dictionary, label = label, folder = folder) 
			if self.figure is not None:
				self.figure_gray = cv2.cvtColor(self.figure, cv2.COLOR_BGR2GRAY)
			try:
				self.tags = {"image_url": self.readFileFromFolder(dir_dictionary, label='image_urls', folder=None),
							 "DOI": self.readFileFromFolder(dir_dictionary, label="dois", folder=None),
							 "label": self.readFileFromFolder(dir_dictionary, label="label", folder=None),
							 "paper_idx": self.readFileFromFolder(dir_dictionary, label="paper_idx", folder=None)}
				# self.tags = {"image_url": self.readFileFromFolder(dir_dictionary, label='image_urls', folder=None),
				# 			 "article_title": self.readFileFromFolder(dir_dictionary, label='title', folder=None)}
			except KeyError:
				raise KeyError("Check to make sure figure tags are correct")
		else:
			self.figure = None
			self.figure_gray = None
			self.tags = None

		# initialize fingerprint dictionary
		self.images = None
		self.Fingerprint = {"figure_tags": [], # tags containing the article title and figure url
							"image_idx": [], # index of image in database
							"image_locs": [], # image location in figure 
							"blot_locs": [], # blot location in image

							"feature_vectors": [], # feature vector (HOG descriptor)

							# "theta": [], # rectifying angle of segmented blot
							# "resolution": [], # original resolution of segmented blot
							# "centroid": [] # center of mass of segmented blot\
							# "contours": [],
							"orientation": [],

							#"reference_label": []
					   		}



	def readFileFromFolder(self, dir_dictionary, label, folder=None):
		"""
		Picks out file from folder as directed by direction dictionary and label. Works with .json file formats
		 used to store .jpeg image url dictionaries from scrapy, the web-scrapping framework.

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
				# load in image if label is 'image' 
				file = cv2.imread(config.dataset_dir + '/' + folder + '/{}'.format(dir_dictionary[label][0]["path"])) 
			except IndexError:
				return(None)
		else:
			file = dir_dictionary[label] # otherwise return the dictionary element requested by key

		if file is not None:
			return(file) # returns file if file exists
		else:
			pass # otherwise, continue searching for other files

	def westernBlotExtractor(self, VISUALIZE=True):
		""" 
		Extracts western blot images from scientific figures

		Parameters:
		-------------
		VISUALIZE: boolean (default: False)
			if true, outputs ploting functions to show image segmentation results

		"""
		## ============================================================================================================
		## ========================= Pre-process the figure ===========================================================
		## ============================================================================================================
		if self.figure is None:
			self.Fingerprint = None
			return(None)
 
		# thresholds image to make contour detection easier 
		thresh_fig = cv2.threshold(self.figure_gray.copy(), config.fig_panel_threshold, 255, cv2.THRESH_BINARY_INV)[1] 

		## ============================================================================================================
		## ========================= Extract images from figure =======================================================
		## ============================================================================================================

		# find contours in the figure
		contours_fig, hierarchy_fig = cv2.findContours(thresh_fig, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:] 
		# approximate each contour to extract only rectangles
		self.images = utils.approxAndExtractRect(self.figure, self.clf_image, contours_fig, hierarchy_fig, 
					 							epsilon=config.cnt_arc_len_approx, lwr_size_lim=config.cnt_area_lwr, 
					 							upr_size_lim=config.cnt_area_upr) 
		#self.visualizeWbImages(self.figure, thresh_fig, contours_fig, self.images, figsize=(6,6))

		## ============================================================================================================
		## ========================= Extract fingerprints from images =================================================
		## ============================================================================================================

		if self.images is not None:
			# visulaize results
			if VISUALIZE is True:
				self.visualizeWbImages(thresh_fig, contours_fig, self.images["grays"], figsize=(6,6))

			# loop over each image in figures:
			for _, image in enumerate(self.images["images"]):
				# call watershed method to collect blot descriptors and locations
				fingerprint, blot_locs = self.blotExtractor(image, VISUALIZE=False)

				if fingerprint is None or blot_locs is None:
					continue
				# check if fingerprints were extracted or if there are too many 
				if len(fingerprint["descriptor"]) != 0 and len(fingerprint["descriptor"]) \
				<= config.max_descriptor_per_image:
					# append the blot data to self.Fingerprint dictionary
					self.appendBlotData(_, blot_locs, fingerprint)
			
			# stack the lists as arrays
			self.Fingerprint["image_locs"] = np.array(self.Fingerprint["image_locs"])

			if len(self.Fingerprint["feature_vectors"]) >= 1:
				for key in ['blot_locs', 'feature_vectors']: #, 'theta', 'resolution', 'centroid']:
					self.Fingerprint[key] = np.vstack(self.Fingerprint[key])
			else:
				for key in ['blot_locs', 'feature_vectors']: #, 'theta', 'resolution', 'centroid']:
					self.Fingerprint[key] = np.array(self.Fingerprint[key])


			# assert that lengths line up
			assert len(self.Fingerprint["image_locs"]) == len(self.Fingerprint["feature_vectors"])
			assert len(self.Fingerprint["figure_tags"]) == len(self.Fingerprint["feature_vectors"])
		else:
			self.Fingerprint = None

	def visualizeWbImages(self, thresh, contours, images, figsize):
		"""
		visualize the western blot figure segmentation method. Plots original figure 
		"""
		copy = self.figure.copy()

		if images is not None:
			for i, image_loc in enumerate(images['loc']):
				x,y,w,h = image_loc
				cv2.rectangle(copy, (int(x), int(y)), (int(x+w), int(y+h)), color=(255,0,0), thickness=3)

		plt.figure(figsize=(20,20))
		plt.imshow(copy)
		plt.savefig("instance_testing/clf_CNN/results/" + str(np.random.randn(1)) + '.png')

		

	def appendBlotData(self, index, locs, fingerprint):
		"""
		appends blot data to Fingerprint dictionary for a given index
		"""

		self.Fingerprint['figure_tags'] += [self.tags]*fingerprint['descriptor'].shape[0]
		self.Fingerprint['image_idx'] += [index]*fingerprint['descriptor'].shape[0]
		self.Fingerprint['image_locs'] += [self.images['loc'][index]]*fingerprint['descriptor'].shape[0]
		self.Fingerprint['blot_locs'].append(locs['loc'])
		self.Fingerprint['feature_vectors'].append(fingerprint['descriptor'])
		# self.Fingerprint['theta'].append(locs['theta'])
		# self.Fingerprint['resolution'].append(locs['resolution'])
		# self.Fingerprint['centroid'].append(locs['centroids'])
		# self.Fingerprint['reference_label'] += fingerprint['reference_label']
		# self.Fingerprint["contours"] += fingerprint["contours"]
		self.Fingerprint["orientation"] += locs["orientation"]

	def blotExtractor(self, image, VISUALIZE=False):
		"""
		Parameters:
		---------------
		image: numpy array
			western blot image to be segmented
		VISUALIZE: bool (default: False)
			if True, outputs visualization figures
		EQUALIZE: bool (default: False)
			if True, extracted blot historgrams will be equalized

		Returns:
		---------------
		fingerprint: dict
			python dictionary containing the blot descriptor
		locs: dict
			python dictionary containing indexing and location information

		"""

		## ============================================================================================================
		## ========================================= Pre-processing ===================================================
		## ============================================================================================================

		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# masked border to eliminate edge effects
		border = self.createBorderMask(image_gray.shape, border_size=config.mask_border_size)
		# invert the image if westernblot image has dark background. 
		if np.mean(image_gray) < config.inv_thresh_value: 
			image_gray = cv2.bitwise_not(image_gray) 

		# build structuring element
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.se_kern_size,config.se_kern_size)) 

		# Smooth and invert image
		image_smooth = cv2.GaussianBlur(image_gray, (config.se_kern_size, config.se_kern_size), 0) # smooth image
		image_inv = cv2.bitwise_not(image_smooth) # foreground must be 'peaks' in image intensities

		# thresholding and mophological operation
		if config.thresh_type == 'otsu':
			thresh = cv2.threshold(image_smooth, 0, maxValue, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
			thresh = cv2.bitwise_and(thresh, thresh, mask=border)
			opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se)
		elif config.thresh_type == 'adaptive':
			thresh_block_size = self.getShapePortion(image_gray.shape, config.adaptive_thresh_block_size)[1]
			# check to make sure the thresholding block size is odd
			if thresh_block_size % 2 == 0:
				thresh_block_size += 1
			thresh_ad_m = cv2.adaptiveThreshold(image_smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 
												thresh_block_size, config.adaptive_thresh_c)
			thresh_ad_m = cv2.bitwise_and(thresh_ad_m, thresh_ad_m, mask=border)
			closing = cv2.morphologyEx(thresh_ad_m, cv2.MORPH_CLOSE, se)
			opening = cv2.morphologyEx(thresh_ad_m, cv2.MORPH_OPEN, se)
		else:
			raise NameError(config.thresh_type + " is not a recognized threshold type")
			
		# build convex hull mask of objects in thresholded image
		mask = self.createConvexMask(cv2.dilate(opening, se, iterations=2)) # masks the local minima to convex hull area
		if mask is None:
			return(None, None)

		## ============================================================================================================
		## ================================= Segmenting Image From Local Minima =======================================
		## ============================================================================================================

		# define local minima parameters
		T = self.getT(image_inv, alpha=config.cut_T, mask=opening) # set max intensity threshold for local min
		order = self.getShapePortion(image.shape, alpha=config.cut_order) # set order for local min detection
		dist_thresh = self.getShapePortion(image.shape, alpha=config.cut_min_spacing)
		vote_lim = self.getShapePortion(image.shape, alpha=config.cut_min_vote) 

		# find local minima location given parameters
		h_coord, v_coord = utils.gridLocalMin(image_inv, T, order) # find local minima coordinates

		# find divide lines given local minima
		divide_line_image = utils.gridLineDetector(opening, h_coord, vote_lim=vote_lim[0], dist_thresh=dist_thresh[1],
												   line_type = 'h', kernel_size=config.se_kern_size, 
												   alpha = config.cut_line_decision, mask=mask) 
		# divide_line_image = utils.gridLineDetector(divide_line_image, v_coord, vote_lim=vote_lim[1], 
		# 											dist_thresh=dist_thresh[0], line_type = 'v', 
		# 											kernel_size=config.se_kern_size, alpha = config.cut_line_decision,
		# 											mask=mask)

		# TODO: combine 'h' and 'v' portions into one function

		## ============================================================================================================
		## ================================= Performing Watershed Operation ===========================================
		## ============================================================================================================


		# find markers
		dtwImage, dtImage = utils.dtwGradient(image_smooth, divide_line_image, config.dist_transform_weight) 
		markers = utils.findMarkers(dtImage, divide_line_image) # starting points for watershed operation

		# define sure foreground, sure background and unkown region to be determined by watershed algorithm
		sure_fg = np.zeros_like(image_gray, dtype=np.uint8) 
		sure_fg[markers>0] = 255 # Sure foreground (i.e. we are sure this is an object)
		sure_bg = cv2.dilate(opening, se, iterations=2) # Sure background (i.e. we are sure this is not an object)
		unknown = cv2.subtract(sure_bg, sure_fg) # unknown region where watershed lines will be placed
		markers = markers + 1
		markers[unknown==255] = 0

		# watershed  
		labels = cv2.watershed(dtwImage, markers.copy())


		## ============================================================================================================
		## ========================================== Visualization ===================================================
		## ============================================================================================================

		if VISUALIZE:

			# # surface plot visualization 
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


			local_min_image = image.copy()
			if h_coord is not None:
				local_min_image[h_coord] = (255,0,0)
			if v_coord is not None:
				local_min_image[v_coord] = (0,0,255)

			segmented_image = image.copy()
			segmented_image[labels==-1] = (255,0,0)

			# Preprocessing images
			f1, a1 = plt.subplots(2,2, figsize = (18,10))
			a1[0,0].imshow(image, "gray"), a1[0,0].set_title("Original Image")
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

			plt.show()


		return(self.appendFingerprintData(image_gray, labels, start_idx=2))

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


	def appendFingerprintData(self, image, markers, start_idx):
		"""
		appends descriptor and location information to fingerprint and locs dictionaries
		"""
		# initialize feature-vector information
		fingerprint = {"image": image.copy(),
					   "blots": [],
					   "descriptor": [],
					   "reference_label": [],
					   "contours": []
					  }

		# initial location information
		locs = {"centroids": [],
				"loc": [],
				"contours": [],
				"theta": [],
				"resolution": [],
				"orientation": []
			   }

		# sift implementation
		if config.SIFT_implement:
			sift = cv2.xfeatures2d.SIFT_create()
			kp, des = sift.detectAndCompute(image,None)

			if des is not None:
				if len(des) > config.max_descriptor_per_image:
					des = des[:config.max_descriptor_per_image]
					kp = kp[:config.max_descriptor_per_image]
				fingerprint["descriptor"] = des 
				locs["loc"] = np.array([key.pt for key in kp])


			return fingerprint, locs


		mag_full, theta_full = utils.sobel_grad(image, ABSOLUTE=True)
		for mark in np.unique(markers)[start_idx:]:

			# ========================================================================================================
			# ==================================================== Contours ==========================================
			# ========================================================================================================

			# Create a mask of the selected blot
			mask = np.zeros(image.shape, dtype=np.uint8)
			mask[markers==mark] = 255

			cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2] # find contour of the blot

 			# contours must not be an empty list
			if len(cnts) < 1:
				continue

			cnt_max = max(cnts, key = cv2.contourArea) # use maximum area contour

			# openCV requires the points in a contour to be above a certain threshold 
			if len(cnt_max) < config.filter_cnt_pts:
				continue

			x,y,w,h = cv2.boundingRect(cnt_max) # extract blot region of interest for viewing purposes
			
			#filter for too-small or too-large of sizes
			if w <= config.pixel_min*2 or h <= config.pixel_min:
				continue 
			if w >= config.pixel_max or h >= config.pixel_max:
				continue
			if (h / w) <= config.ar_min or (h / w) >= config.ar_max:
				continue

			# fit ellipse to compute centroid angle and major and minor principal axes
			(cx, cy), (a1, a2), theta = cv2.minAreaRect(cnt_max)  # fit ellipse
			if a1 > a2:
				MA = a1
				ma = a2
			else:
				theta = 90+theta
				ma = a1
				MA = a2
				


			# =========================================================================================================
			# ==================================================== Orientation ========================================
			# =========================================================================================================
			e = (int(cx+1), int(cy+1)), (int(MA/2), int(ma/2)), int(theta)
			dd, dc = utils.dd_dc_ellipse(image, e)
			rotation_matrix = cv2.getRotationMatrix2D((cx, cy), theta, 1) 
			rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0])) 
			block_y, block_x = utils.pickBlockSize(ma, MA-config.ellipse_corr, (4,4)) 
			oriented_blot = self.isolateRotatedBox(rotated_image, cx, block_x, cy, block_y)

			# mag_blot, _ = utils.sobel_grad(oriented_blot, ABSOLUTE=True)
			dd, dc = utils.dd_dc_box(oriented_blot)

			if dd < 0:
				oriented_blot = np.rot90(oriented_blot, 2)
			if dc < 0:
				oriented_blot = np.flip(oriented_blot, axis=0)

			locs["orientation"].append([(cx,cy), (ma, MA), theta, (dd, dc)])	

			# plt.figure()
			# plt.imshow(oriented_blot, "gray")
			# print(dd,dc)
			# plt.show()	



			# =========================================================================================================
			# ==================================================== Filtering Operation ================================
			# ========================================================================================================= 

			# equalize image if called
			# if EQUALIZE is True:
			# 	oriented_blot = cv2.equalizeHist(oriented_blot)
			# extracted blot must be 2-dimensional
			if oriented_blot.shape[0] == 0 or oriented_blot.shape[1] == 0: 
				continue
			# filter for basic image statistics
			if np.mean(oriented_blot) >= np.mean(image):
				continue # guarantees the blot is a foreground object
			if np.all(np.var(image[y:y+h, x:x+w], axis=1) <= config.variance_lim) \
			   or np.all(np.var(image[y:y+h, x:x+w], axis=0) <= config.variance_lim):
				continue # avoids approximately uniform stripes

			#TODO: make the "window-stretch" option extend into the entire figure space not just limited to the image
			r1, r2, r3, r4 = utils.get_rect(x,y,w,h,image.shape, config.window_stretch)
			perc_white = np.sum(image[r1:r3, r2:r4] == 255) / np.size(image[r1:r3, r2:r4])
			if perc_white >= config.white_max:
				continue

			#filter for higher-order statistics
			classification = utils.blotFilter(image,x,y,w,h, self.clf_blot)
			if classification == 0:
				continue

			# =========================================================================================================
			# ==================================================== Feature Vector======================================
			# ========================================================================================================= 

			# compute blot descriptor: histogram of oriented gradients (magnitude and direction)
			des = utils.HOG(oriented_blot,
							binShape = config.HOG_bin_shape, 
							orientations = config.HOG_orientations, 
							L2_NORMALIZE = True)

			# descriptor must not be None object
			if des is None:
				continue

			# append feature descriptor information
			fingerprint["descriptor"].append(des)
			fingerprint["blots"].append(oriented_blot)
			#fingerprint["contours"].append(cnt_max)
			#fingerprint["reference_label"].append(reference_label)

			# append feature location information
			locs["loc"].append((x,y,w,h))
			locs["contours"].append(cnt_max)
			locs["centroids"].append((cx,cy)) 
			locs["theta"].append([theta])
			locs["resolution"].append((ma,MA))

		# convert to numpy array to avoid compatability issues
		fingerprint["descriptor"] = np.array(fingerprint["descriptor"])
		for key in ["loc", "contours", "centroids", "theta", "resolution"]:
			locs[key] = np.array(locs[key])

		return(fingerprint, locs)

	def isolateRotatedBox(self, rotated_blot, cx, block_x, cy, block_y ):
		"""
		isolates blot from rotated image and returns truncated array
		"""
		(cx, cy) = (int(round(cx)),int(round(cy))) # make sure int values

		# compute bounds for bounding box
		lbx = np.maximum(int(round(cx - (block_x/2) +1)), 0) # lower and upper x-axis bounds for bounding rectangle
		ubx = np.maximum(int(round(cx + (block_x/2) -1)), 0) 
		lby = np.maximum(int(round(cy - (block_y/2) +1)), 0) # lower and upper y-axis bounds for bounding rectangle
		uby = np.maximum(int(round(cy + (block_y/2) -1)), 0) 

		return(rotated_blot[lby:uby, lbx:ubx]) # extract rotated blot from image



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


def collect_fingerprints(TO_FILE, FOLDER_NAME, TO_FINGERPRINT, DATASET_TYPE):

	## ================================================================================================================
	## =========================================Load in western blot data==============================================
	## ================================================================================================================
	if DATASET_TYPE == "ref":
		data = np.load(TO_FILE)
		sess = tf.InteractiveSession()
		#clf_image = wb_clf(clf_image_filename, input_dim=(50,100), reshape_size=13*25*64, name="image")
		clf_image = wb_clf(config.clf_image_filename, reshape_size=None, input_dim=(100,200), name="image", 
						   ALEXNET=True)
		clf_blot = wb_clf(config.clf_blot_filename, input_dim=(15,30), reshape_size=4*8*64, name="blot")
		sess.run(tf.global_variables_initializer())
	else:
		raise NameError("There is no DATASET_TYPE: %s"%(DATASET_TYPE))
	
	it_tracker = 0
	print(utils.print_time(time.time()), ': starting fingerprint extraction process')
	# loop over all directory dictionaries housing paths to western blot figures and metadata
	for _, dir_dictionary in enumerate(data):
		## ============================================================================================================
		## ========================================Create WesternBlot object===========================================
		## ============================================================================================================ 
		if DATASET_TYPE == "ref":
			WB = WesternBlot(dir_dictionary, "images", FOLDER_NAME, clf_image, clf_blot)
		if WB.figure is None:
			continue
		## ============================================================================================================
		## =========================================Extract fingerprints+==============================================
		## ============================================================================================================

		WB.westernBlotExtractor(VISUALIZE=False)
		if WB.Fingerprint is not None:
			if np.size(WB.Fingerprint["feature_vectors"]) != 0:
				if it_tracker == 0:
					Fingerprints = WB.Fingerprint 
					Fingerprints["figure_number"] = [_]*WB.Fingerprint["feature_vectors"].shape[0] 
				else:
					# loop over keys in Fingerprint dictionary and append descriptors and metadata
					for key in WB.Fingerprint.keys():
						try:
							if key == "figure_tags" or key == "image_idx" or key == "reference_label": 
							 	Fingerprints[key] += WB.Fingerprint[key]
							else:
								if WB.Fingerprint[key].shape[0] is None:
									WB.Fingerprint[key].reshape(1,np.size(WB.Fingerprint[key]))
								Fingerprints[key] = np.vstack((Fingerprints[key], WB.Fingerprint[key]))
						except ValueError:
							print("Figure %r had the problem fingerprint and key: "
								%(_+1), len(WB.Fingerprint[key]), key)
							break
						except AttributeError:
							print("Figure %r had the problem fingerprint and key: "
								%(_+1), len(WB.Fingerprint[key]), key)
							break

					# append figure indexes 
					Fingerprints["figure_number"] += [_]*WB.Fingerprint["feature_vectors"].shape[0]
				it_tracker += 1
			else:
				pass
				#print("There were no feature vectors in the images of figure %s"%(_+1))
		else:
			pass
			#print("There were no images in figure %s"%(_+1))

		# print progress report to command line every 10 figures
		if (_ + 1) % 10 == 0:
			print(utils.print_time(time.time()), "|| Analyzed %r 0f %r figures : %r total feature vectors extracted"
				%(_+1, len(data), Fingerprints["feature_vectors"].shape[0]))

	assert Fingerprints["feature_vectors"].shape[0] == len(Fingerprints["figure_number"])

	# save fingerprint in .npy format
	np.save(TO_FINGERPRINT, Fingerprints)

	# print final progress report to command line
	print(utils.print_time(time.time()), "|| Completed. %r figures used and %r features extracted"% 
		(len(np.unique(Fingerprints["figure_number"])), Fingerprints["feature_vectors"].shape[0]))









