import numpy as np 
import cv2
import matplotlib.pyplot as plt
import wbFingerprintUtils as utils
import json
from scipy import ndimage as ndi 
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
	  # Filter out impossible shapes (i.e very small width and/or length for blots)

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
	print("There Are %r images in this set"%(len(dir_dictionary)))
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

		if dir_dictionary is not None:
			self.figure = self.readFileFromFolder(dir_dictionary, label = label, folder = folder) # self.figure = cv2.imread(dir_dictionary)
			if self.figure is not None:
				self.figure_gray = cv2.cvtColor(self.figure, cv2.COLOR_BGR2GRAY)
		else:
			self.figure = None
			self.figure_gray = None

		try:
			self.tags = {"image_url": self.readFileFromFolder(dir_dictionary, label='image_urls', folder=None),
						 "article_title": self.readFileFromFolder(dir_dictionary, label='title', folder=None)
						}
		except KeyError:
			pass

		# initialize final fingerprint attribute
		self.Fingerprint = {"image_locs": [], # image location in figure 
							"blot_locs": [], # blot location in image
							"feature_vectors": [], # feature vector (HOG descriptor)
							"figure_tags": [], # tags regarding the article title and figure url
					   		}

		# Global variables
		self.ELLIPSE_CORR = 4 # corrects noise due to ellipse overfitting
		self.BLOT_SIZE_LIM_MIN = 0.005 # set to a fraction of image_gray.size to filter small blots
		self.BLOT_SIZE_LIM_MAX = 0.90
		self.IDEAL_DESCRIPTOR = np.load("ideal_descriptor.npy") # load to filter out none "blot-like" descriptors
		self.L2_LIM = 1 # L2 distance limit for "blot-like" descriptors

	def readFileFromFolder(self, dir_dictionary, label, folder=None):
		"""
		Picks out file from folder as directed by direction dictionary and label

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
					str(input("Input folder name: ")) # if forgot to specify folder type, chance to input it
				except ValueError: 
					print("That was not a valid folder name. Please try again") # if input is not a string, raise error

			#print("testFigures/" + folder + '/{}'.format(dir_dictionary[label][0]["path"]))
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

	def westernBlotExtractor(self,
							 break_point=0.90, 
							 maxValue=255, threshold=75, 
							 alpha=1000, beta=.05, 
							 epsilon=0.18, size_lim=(2.8,1), whr_lim=(2,1), mode_lim=(3,1), shift_length=4, hist_lim=2.0,
							 VISUALIZE=False, max_num_des_per_img = 12
							 ):
		""" 
		Extracts western blot images from scientific figures

		Parameters:
		-------------
		break_point: float (default = 0.90)
			defines the contrast threshold for ectified function
		maxValue: uint8 (default = 255)
			The maximum value in thresholding function
		threshold: uint8 (default = 75)
			defines the thresholding value in thresholding function
		alpha: float (default = 1000)
			gives maximum pixel area of contour
		beta: float (default = .05)
			gives minimum cutoff of contour area as percentage of max area
		epsilon: float (default = 0.18)
			percentage of arc length to approximate contour
		size_lim: tuple 
			determines the number of standard deviations (above, below) to set size limit from mean
		whr_lim: tuple
			determines the number of standard deviations (above, below) to set width/height ratio limit from mean
		mode_lim: tuple
			determines the number of standard deviations (above, below) to set mode limit from mean
		shift_length: int (default = 4):
			gives the bin length from mode to compare against
		hist_lim: float (default = 2.0)
			gives the scale multiple limit for whcih the mode can be from its neighbors

		Returns:
		-------------
		"""


		cFiltered = utils.contrastFilters(self.figure_gray, break_point = break_point) # apply a filter to exagerate contrast between mid-level grayvalues and white  
		thresh = cv2.threshold(cFiltered, threshold, maxValue, cv2.THRESH_BINARY_INV)[1] # thresholds image to make contour detection easier
		contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] # find contours in the figure

		# filter contours by size (i.e. discard contours too small or too large)
		copy = self.figure.copy()
		contours = utils.filterContourArea(contours, np.size(self.figure_gray), alpha=alpha, beta=beta) 

		# approximate each contour to extract only rectangles
		images = utils.approxAndExtractRect(self.figure, contours, 
											epsilon=epsilon, 
											size_lim=size_lim, 
											whr_lim=whr_lim,
											mode_lim=mode_lim, shift_length=shift_length, hist_lim=hist_lim 
											)
		if images is not None:

			if VISUALIZE is True:
				plt.figure()
				plt.imshow(thresh, "gray")
				cv2.drawContours(copy, contours, -1, (0,0,255), 2)
				plt.figure()
				plt.imshow(copy, "gray")
				for i, image in enumerate(images["grays"]):
					f,a = plt.subplots(1,2)
					a[0].imshow(image, "gray")
					a[1].hist(image.flatten(), 256, [0,256], color = 'r')
					a[1].set_title(str(np.mean(image.flatten())) + ", " + str(np.var(image.flatten())))
					a[1].set_xlim([0,255])
				plt.show()

			# loop over each image in figures:
			none_check = 0
			for _, image in enumerate(images["images"]):
				fingerprint, locs = self.blotExtractorWatershed(image, maxValue=255, WATERSHED="openCV", VISUALIZE=False)
				if len(fingerprint["blots"]) != 0 and len(fingerprint["blots"])<=max_num_des_per_img:
					none_check += 1
					if none_check == 1:
						if locs["loc"].shape[0] is None:
							locs["loc"].reshape(1, np.size(locs["loc"]))
							print("Reshaped!")
						if fingerprint["descriptor"].shape[0] is None:
							fingerprint["descriptor"].reshape(1, np.size(fingerprint["descriptor"]))
							print("Reshaped!")

						self.Fingerprint["figure_tags"] = [self.tags]*len(fingerprint["descriptor"])
						self.Fingerprint["image_locs"] = [images["loc"][_]]*len(fingerprint["descriptor"])
						self.Fingerprint["blot_locs"] = locs["loc"]
						self.Fingerprint["feature_vectors"] = fingerprint["descriptor"]
					else:
						if locs["loc"].shape[0] is None:
							locs["loc"].reshape(1, np.size(locs["loc"]))
						if fingerprint["descriptor"].shape[0] is None:
							fingerprint["descriptor"].reshape(1, np.size(fingerprint["descriptor"]))

						self.Fingerprint["blot_locs"] = np.vstack((self.Fingerprint["blot_locs"], locs["loc"]))
						self.Fingerprint["feature_vectors"] = np.vstack((self.Fingerprint["feature_vectors"], fingerprint["descriptor"]))
						self.Fingerprint["figure_tags"] += [self.tags]*fingerprint["descriptor"].shape[0]
						self.Fingerprint["image_locs"] += [images["loc"][_]]*fingerprint["descriptor"].shape[0] 

					#self.imshowBlots(fingerprint["blots"], title=str(_))
			# convert each to an array
			self.Fingerprint["image_locs"] = np.array(self.Fingerprint["image_locs"])

			if len(self.Fingerprint["image_locs"]) != len(self.Fingerprint["feature_vectors"]):
				print("The lengths of the dictionary lists should be equal, something went wrong")
			if len(self.Fingerprint["figure_tags"]) != len(self.Fingerprint["feature_vectors"]):
				print("The lengths of the dictionary lists should be equal, something went wrong")

		else:
			self.Fingerprint = None

		
	def blotExtractorWatershed(self, image, maxValue, inv_thresh=100, kernelSize=(3,3), num_bins=256, 
							   LOCAL_MIN_BORDER = True, WATERSHED="g-w-DT", VISUALIZE=False, EQUALIZE=False):
		"""
		Extracts blot components from a western blot image using watershed method. TODO: Normalizes orientation, scale, and contrast. 

		Parameters:
		------------
		inv_thresh: int (default: 100)
			threshold for average pixel value to invert image or not.  
		maxValues: int
			sets the pixel value for the thresholded image
		kernelSize: tuple (default: (5,5))
			sets the kernel size for denoising and dilation 
		num_bins: int (default: 256)
			sets the number of bins for extracting the cdf of each blot
		LOCAL_MIN_BORDER: boolean (default: True)
			draws borders seperating blots based on local minima detection
		WATERSHED: str (default="openCV")
			string to decide watershed method
		VISUALIZE: boolean (default: False)
			visualize the watershed process
		EQUALIZE: boolean (default: False)
			equalize the rotated_blots

		Returns:
		------------
		fingerprint: dictionary
			comprised of the original gray-scale image, segmented blot units, and corresponding HOG descriptor
		locs: dcitionary
			comprised of centroids, orientation angles, and coordinate locations for each corresponding blot unit
		"""
		# convert image to grayscale
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if np.mean(image_gray) < inv_thresh: 
		 	image_gray = cv2.bitwise_not(image_gray) # invert thresh if westernblot image is inverted.
				
		# initialize fingerprint
		fingerprint = {"image": image_gray,
					   "blots": [],
					   "descriptor": []
					  }

		# initial location information
		locs = {"centroids": [],
				"angles": [],
				"loc": []
			   }

		if LOCAL_MIN_BORDER is True:
			image_gray_inv = cv2.bitwise_not(image_gray)
			fit_lines, image_gray_inv = utils.detectLocalMinima(image_gray_inv, k=5, g_k=9, T=65, w=2, dub=15, c=300, slope_angle=(75,2), order=1, VISUALIZE=False)
			image_gray = cv2.bitwise_not(image_gray_inv)

		# Threshold image
		thresh = cv2.threshold(image_gray, 0, maxValue, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

		# noise removal
		kernel = np.ones(kernelSize, dtype=np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

		# watershed using openCV
		if WATERSHED == "openCV":
			# Find sure background area
			sure_bg = cv2.dilate(opening, kernel, iterations=1)

			# Find sure foreground area
			dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) # distance transform to find sure foreground
			dist_thresh = utils.distThresh(dist_transform, 20) # must find a threshold that maximizes number of sure regions
			sure_fg = np.uint8(cv2.threshold(dist_transform, dist_thresh*dist_transform.max(), 255, 0)[1])
			#sure_fg = utils.adaptiveDilation(sure_fg)
			#sure_fg = # np.uint8(cv2.dilate(sure_fg, kernel, iterations=2))#cv2.morphologyEx(np.uint8(sure_fg), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

			# Find unknown region
			unknown = cv2.subtract(sure_bg, sure_fg) 

			# Label the markers
			markers = cv2.connectedComponents(sure_fg)[1]
			markers = markers+1 # add one to all labels so that sure background is not 0, but 1

			# Perform watershed method to segment the blots
			markers[unknown==255] = 0 # mark the region of unknown with zero
			blur = cv2.GaussianBlur(image, kernelSize, 0)
			markers = cv2.watershed(blur, markers)
			start_idx = 2

		# watershed using scipy.ndimage and skimage
		elif WATERSHED == "scipy":
			dist_transform = ndi.distance_transform_edt(opening)
			local_max = peak_local_max(dist_transform, indices=False, labels=opening)
			markers = ndi.label(sure_fg)[0]
			markers = watershed(-dist_transform, markers, mask=opening)
			start_idx = 1

		elif WATERSHED == "g-w-DT":
			markers = utils.gwWatershed(image_gray, md_k=5, se_k=5, g_k=5, grad_weigh=2, maxValue=255, VISUALIZE=False)
			start_idx = 1

		else:
			raise NameError(WATERSHED + " is not a recognized watershed type")

		# Visualizetion
		if VISUALIZE is True:
			if WATERSHED == "openCV":
				f, ax = plt.subplots(3,2, figsize=(9,5))
				ax[0,0].imshow(thresh, "gray"), ax[0,0].set_title("Original Thresholded Image"), ax[0,0].axis("off")
				ax[1,0].imshow(sure_bg, "gray"), ax[1,0].set_title("Sure Background"), ax[1,0].axis("off")
				ax[2,0].imshow(dist_transform, "gray"), ax[2,0].set_title("Distance Transform"), ax[2,0].axis("off")
				ax[0,1].imshow(sure_fg, "gray"), ax[0,1].set_title("Sure Foreground"), ax[0,1].axis("off")
				ax[1,1].imshow(unknown, "gray"), ax[1,1].set_title("Unkown Region: Define Boundaries Here"), ax[1,1].axis("off")
				ax[2,1].imshow(markers), ax[2,1].set_title("Updated markers"), ax[2,1].axis("off")
			elif WATERSHED == "scipy":
				f, ax = plt.subplots(2,2, figsize=(9,5))
				ax[0,0].imshow(thresh, "gray"), ax[0,0].set_title("Original Thresholded Image"), ax[0,0].axis("off")
				ax[1,0].imshow(dist_transform, "gray"), ax[1,0].set_title("Distance Transform"), ax[1,0].axis("off")
				ax[0,1].imshow(sure_fg, "gray"), ax[0,1].set_title("Local Max"), ax[0,1].axis("off")
				ax[1,1].imshow(markers, "gray"), ax[1,1].set_title("Watershed Markers"), ax[1,1].axis("off")
			else:
				raise NameError(WATERSHED + " is not a recognized watershed type")
			plt.tight_layout()

		# Extract features for each blot:
		for mark in np.unique(markers)[start_idx:]:

			# Create a mask of the selected blot
			mask = np.zeros(image_gray.shape, dtype=np.uint8)
			mask[markers==mark] = 255

			# find contour of the blot
			cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
			cnt_max = max(cnts, key = cv2.contourArea)

			if len(cnt_max) < 5: # Must have at least 5 points to fit ellipse
				continue

			# extract blot region of interest
			x,y,w,h = cv2.boundingRect(cnt_max)

			# fit ellipse to compute orientation angle and major and minor principal axes
			(cx, cy), (ma, MA), theta = cv2.fitEllipse(cnt_max) # fit ellipse
			locs["centroids"].append((cx,cy)) # append centroids for further analysis
			(cx, cy) = (int(round(cx)),int(round(cy))) # make sure int values
			block_y, block_x = utils.pickBlockSize(ma, MA-self.ELLIPSE_CORR, (4,4)) # make sure block shape is multiple of 4 for resampling
			theta = theta - 90 # opencv likes theta rotated relative to 90 degrees
			locs["angles"].append(theta) # append angles for further analysis

			# normalize orientation of blot by rotating about centroid
			rotation_matrix = cv2.getRotationMatrix2D((cx, cy), theta, 1) # define the rotation matrix for angle theta
			rotated_blot = cv2.warpAffine(image_gray, rotation_matrix, (image_gray.shape[1], image_gray.shape[0])) # rotate the image by angle theta

			# compute bounds for bounding box
			lbx = np.maximum(int(round(cx - (block_x/2))), 0) # lower and upper x-axis bounds for bounding rectangle
			ubx = np.maximum(int(round(cx + (block_x/2))), 0) 
			lby = np.maximum(int(round(cy - (block_y/2))), 0) # lower and upper y-axis bounds for bounding rectangle
			uby = np.maximum(int(round(cy + (block_y/2))), 0) 

			rotated_blot = rotated_blot[lby:uby, lbx:ubx] # extract rotated blot from image
			if EQUALIZE is True:
				rotated_blot = cv2.equalizeHist(rotated_blot)

			if rotated_blot.size < self.BLOT_SIZE_LIM_MIN * image_gray.size or abs(w*h) > self.BLOT_SIZE_LIM_MAX * image_gray.size: #  TODO: set area parameters less arbitrarily
				continue
			# compute blot gradient (magnitude and direction)
			des = utils.HOG(rotated_blot, blockShape = (6,24), binShape = (2,4), orientations = 8, L2_NORMALIZE = True)

			if des is None:
				continue
			if np.linalg.norm(self.IDEAL_DESCRIPTOR - des) > self.L2_LIM:
				continue
			if h/w < 1/8:
				continue

			# append blots and features to lists
			fingerprint["descriptor"].append(des)
			fingerprint["blots"].append(rotated_blot)
			locs["loc"].append((x,y,w,h))

		# convert to numpy array
		fingerprint["descriptor"] = np.array(fingerprint["descriptor"])
		locs["loc"] = np.array(locs["loc"])

		return(fingerprint, locs)


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
	pathToFile = "testFigures/nature_wb_output/output_revised.json"
	data = loadWesternBlotData(pathToFile)

	# load in Fingerprints
	pathToFingerprints = "testFigures/nature_wb_output/fingerprintDatabase.npy"
	Fingerprints = np.load("testFigures/nature_wb_output/fingerprintDatabase.npy").item()

	for i in idx_list:
		if i < 0 or i > Fingerprints["feature_vectors"].shape[0]:
			raise IndexError("an index in idx_list is out of bounds for Fingerprint with length: ", Fingerprints["feature_vecors"].shape[0])

		figure_idx = Fingerprints["figure_idx"][i]
		image_loc = Fingerprints["image_locs"][i]
		blot_loc = Fingerprints["blot_locs"][i]
		#print(blot_loc)
		tags = Fingerprints["figure_tags"][i]

		figure = WesternBlot(data[figure_idx], label="images", folder="nature_wb_output").figure_gray
		image = figure[image_loc[1]:image_loc[1]+image_loc[3], image_loc[0]:image_loc[0]+image_loc[2]]
		blot = image[blot_loc[1]:blot_loc[1]+blot_loc[3], blot_loc[0]:blot_loc[0]+blot_loc[2]]

		#print(figure.shape)

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

def main():
	## load in western blot data
	pathToFile = "testFigures/nature_wb_output/output_revised.json"
	data = loadWesternBlotData(pathToFile)
	#data = np.load("testData/test_westernblots.npy") # need to bring in as list of jpeg files

	none_check = 0
	time.clock()
	for _, dir_dictionary in enumerate(data[:10]):
		WB = WesternBlot(dir_dictionary, label="images", folder="nature_wb_output")
		if WB.figure is None:
			continue
		WB.westernBlotExtractor()
		if WB.Fingerprint is not None:
			if np.size(WB.Fingerprint["feature_vectors"]) != 0:
				none_check+=1
				if none_check == 1:
					Fingerprints = WB.Fingerprint
					Fingerprints["figure_idx"] = [_]*WB.Fingerprint["feature_vectors"].shape[0]
				else:
					for key in WB.Fingerprint.keys():
						try:
							if key != "figure_tags": 
								if WB.Fingerprint[key].shape[0] is None:
									WB.Fingerprint[key].reshape(1,np.size(WB.Fingerprint[key]))
								Fingerprints[key] = np.vstack((Fingerprints[key], WB.Fingerprint[key]))
							else:
							 	Fingerprints[key] += WB.Fingerprint[key]
						except ValueError:
							print("Figure %r had the problem fingerprint and key: "%(_+1), len(WB.Fingerprint[key]), key)
							break
						except AttributeError:
							print("Figure %r had the problem fingerprint and key: "%(_+1), len(WB.Fingerprint[key]), key)
							break
					Fingerprints["figure_idx"] += [_]*WB.Fingerprint["feature_vectors"].shape[0]
			else:
				print("There were no feature vectors in the images of figure %s"%(_+1))

		else:
			print("There were no images in figure %s"%(_+1))

		if (_ + 1) % 10 == 0:
			print(round(time.clock()), "secs: Analyzed %r figures: %r total feature vectors extracted"%(_+1, Fingerprints["feature_vectors"].shape[0]))

	if Fingerprints["feature_vectors"].shape[0] != len(Fingerprints["figure_idx"]):
		print("The lengths of the dictionary lists should be equal, something went wrong")

	output_filename = "testFigures/nature_wb_output/fingerprintDatabase.npy"
	np.save(output_filename, Fingerprints)

	print(round(time.clock()), "secs: Completed. %r figures used and %r features extracted"% (len(np.unique(Fingerprints["figure_idx"])), Fingerprints["feature_vectors"].shape[0]))


if __name__ == '__main__':
	#main()
	view(range(60,80), (9,5))


