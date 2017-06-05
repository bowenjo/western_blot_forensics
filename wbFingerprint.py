import numpy as np 
import cv2
import matplotlib.pyplot as plt
import wbFingerprintUtils as utils
import json
from scipy.ndimage import label


"""
CLASS FOR FINGERPRINTING WESTERN BLOT PLOTS FOR IMAGE FORENSICS
"""
##TODO: 
	  #	X for blots with connected portions: Look into a more refined object recognition algorithm (watershed segmentation).
	  # refine area selection/filtering in utils.
	  # perhaps instead of raw pixels, take slices through center of mass (along principal axes defined by ellipse). Rotation invariance, no need to normalize orientation.
	  # taking slices could also be scale invariant if slice distribution is normalized/approximated
	  # add in work around to None blots if they occur.
	  # X design a better equalization function that ignores black background(utils). 
	  # decide which angle measurement is better: from moments or from fitted ellipse
	  # X correct so will work if blots are originally white (inverse images) <--did not change for blot extractor original

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
	figure_data = json.loads(open(filename).read())
	return(figure_data)


class WesternBlot(object):
	""" 
	Class that encapsulates fingerprinting western blot images for digital forensics. To be performed on single, isolated western blot images. 
	Creates a dictionary containing pixel information and statistical properties for each blot in the image. Elements in the dictionary
	are split into units where each unit is a representation of a single blot. Blot images can be normalized for scale, rotation, and contrast.
	Translational invariance comes from extracting each blot seperately as a single template unit. The class also includes plotting functions
	that can be called after object.blotExtractor() has been executed. 
	"""

	def __init__(self, figure_data=None):
		#self.image = cv2.imread("output/{}".format(figure_data["images"][0]["path"])) # brings .jpeg figures into python environment
		if figure_data is not None:
			self.figure = cv2.imread(figure_data)
			self.figure_gray = cv2.cvtColor(self.figure, cv2.COLOR_BGR2GRAY)
		else:
			self.figure = None
			self.figure_gray = None

		self.location = figure_data #could be title, article url, authors, publisher, etc.

		self.EPSILON = 4 # corrects noise due to ellipse overfitting
		self.EPSILON_TWO = 1/64 # set to a fraction of image_gray.size to filter small blots 

	def westernBlotExtractor(self, wb_only = False, maxValue=255, threshold=75, eta=0.2):
		self.image = None
		self.image_gray = None
		""" 
		Extracts western blot images from scientific figures
		"""
		# equ = cv2.equalizeHist(self.figure_gray) # exagerates borders in the figure making thresholding easier
		# thresh = cv2.threshold(equ, threshold, maxValue, cv2.THRESH_BINARY_INV) # thresholds image to make edge detection easier
		# contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] # find contours in the figure

		# ## filter ontours by size (i.e. discard contours too small or too large)
		# areas = [cv2.contourArea(cnt) for cnt in contours] # all areas
		# min_area  = np.mean(areas) - np.std(areas) # designated minimum area 
		# max_area = np.mean(areas) + np.std(areas) # designated maximum area
		# areas = [a for a in area if a < max_area and a > min_area] # filter areas by designated bounds

		# contours = [cnt for cnt in contours if cv2.contourArea(cnt) in areas] # filter contours by designated bounds

		# ## approximate each contour to extract only rectangles
		# self.images = []
		# for cnt in contours:
		# 	epsilon = eta * cv2.arcLength(cnt, True) # accuracy value a perentage of contour arc length
		# 	approx = cv2.approxPolyDP(cnt, epsilon, True) # approximate the contour to a simpler shape (i.e diagnol line, rectangle, etc.)
		# 	x,y,w,h = cv2.boundingRect(approx) # create bounding rectangle to extract region of interest around approximated contour

		# 	roi = self.figure_gray[y:y+h, x:x+w]
		# 	#TODO: filter rectangular regions of interest to only contain western blot images 
		# 	self.images.append(roi)

		self.image = self.figure
		self.image_gray = self.figure_gray
		

	def blotExtractorWatershed(self, maxValue, inv_thresh=100, kernelSize = (3,3), num_bins=256, Visualize=True):
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
		Visualize: boolean (default: False)
			visualize the watershed process

		Returns:
		------------
		None
		"""
		## intitialize fingerprint
		self.fingerprint = {"blots": [],
							"axes": [],
							"cdfs": [],
							"image": self.image_gray,
							"location": self.location,
							"descriptor": []
							}

		self.centroids_angles = {"centroids": [],
								 "angles": []
								 }

		## Threshold image
		image = self.image.copy()
		thresh = cv2.threshold(self.image_gray, 0, maxValue, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
		# if np.mean(self.image_gray) < inv_thresh: 
		# 	thresh = cv2.bitwise_not(thresh) # invert thresh if westernblot image is inverted.

		## noise removal
		kernel = np.ones(kernelSize, dtype=np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

		## Find sure background area
		sure_bg = cv2.dilate(opening, kernel, iterations=1)

		## Find sure foreground area
		dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) # distance transform to find sure foreground
		dist_thresh = utils.distThresh(dist_transform, 20) # must find a threshold that maximizes number of sure regions
		sure_fg = cv2.threshold(dist_transform, dist_thresh*dist_transform.max(), 255, 0)[1]
		sure_fg = np.uint8(sure_fg)
		#sure_fg = cv2.dilate(sure_fg, kernel, iterations=1) # expand sure foreground to correct for noise

		## Find unknown region
		unknown = cv2.subtract(sure_bg, sure_fg) 

		## Label the markers
		markers = cv2.connectedComponents(sure_fg)[1]

		markers = markers+1 # add one to all labels so that sure background is not 0, but 1
		markers[unknown==255] = 0 # mark the region of unknown with zero

		## Perform watershed method to segment the blots

		blur = cv2.GaussianBlur(image, kernelSize, 0)
		# brightness_array  = np.uint8(0 * np.ones(image.shape))
		# adjContrast_image = cv2.scaleAdd(image, 2, brightness_array)
		# blur = cv2.GaussianBlur(adjContrast_image, kernelSize, 0)
		# plt.figure()
		# plt.imshow(blur, "gray")
		#blur = cv2.pyrMeanShiftFiltering(image, 21, 51)
		#image = cv2.cvtColor(cv2.bitwise_not(opening), cv2.COLOR_GRAY2BGR)
		#markers = cv2.watershed(image, markers)
		markers = cv2.watershed(image, markers)


		## Visualizetion
		if Visualize is True:
			f, ax = plt.subplots(3,2, figsize=(9,5))
			ax[0,0].imshow(thresh, "gray"), ax[0,0].set_title("Original Thresholded Image"), ax[0,0].axis("off")
			ax[1,0].imshow(sure_bg, "gray"), ax[1,0].set_title("Sure Background"), ax[1,0].axis("off")
			ax[2,0].imshow(dist_transform, "gray"), ax[2,0].set_title("Distance Transform"), ax[2,0].axis("off")
			ax[0,1].imshow(sure_fg, "gray"), ax[0,1].set_title("Sure Foreground"), ax[0,1].axis("off")
			ax[1,1].imshow(unknown, "gray"), ax[1,1].set_title("Unkown Region: Define Boundaries Here"), ax[1,1].axis("off")
			ax[2,1].imshow(markers), ax[2,1].set_title("Updated markers"), ax[2,1].axis("off")
			plt.tight_layout()

		## Extract features for each blot:
		for mark in np.unique(markers)[2:]:

			## Create a mask of the selected blot
			mask = np.zeros(self.image_gray.shape, dtype=np.uint8)
			mask[markers==mark] = 255

			## find contour of the blot
			cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
			cnt_max = max(cnts, key = cv2.contourArea)

			## extract blot region of interest
			x,y,w,h = cv2.boundingRect(cnt_max)
			blot = self.image_gray[y:y+h, x:x+w]

			
			if len(cnt_max) > 5: # Must have at least 5 points to fit ellipse

				## fit ellipse to compute orientation angle and major and minor principal axes
				(cx, cy), (ma, MA), theta = cv2.fitEllipse(cnt_max) # fit ellipse
				self.centroids_angles["centroids"].append((cx,cy)) # append centroids for further analysis
				(cx, cy) = (int(round(cx)),int(round(cy))) # make sure int values
				block_y, block_x = utils.pickBlockSize(ma, MA-self.EPSILON, (4,4)) # make sure block shape is multiple of 4 for resampling
				theta = theta - 90 # opencv likes theta rotated relative to 90 degrees
				self.centroids_angles["angles"].append(theta) # append angles for further analysis

				## normalize orientation of blot by rotating about centroid
				rotation_matrix = cv2.getRotationMatrix2D((cx, cy), theta, 1) # define the rotation matrix for angle theta
				rotated_blot = cv2.warpAffine(self.image_gray, rotation_matrix, (self.image_gray.shape[1], self.image_gray.shape[0])) # rotate the image by angle theta

				## compute bounds for bounding box
				lbx = np.maximum(int(round(cx - (block_x/2))), 0) # lower and upper x-axis bounds for bounding rectangle
				ubx = np.maximum(int(round(cx + (block_x/2))), 0) 
				lby = np.maximum(int(round(cy - (block_y/2))), 0) # lower and upper y-axis bounds for bounding rectangle
				uby = np.maximum(int(round(cy + (block_y/2))), 0) 

				rotated_blot = rotated_blot[lby:uby, lbx:ubx] # extract rotated blot from image

				if rotated_blot.size > self.EPSILON_TWO * self.image_gray.size: #  TODO: set area parameters less arbritrarily

					## compute blot gradient (magnitude and direction)
					des = utils.HOG(rotated_blot, blockShape = (6,24), binShape = (2,4), orientations = 8, L2_NORMALIZE = True)

					## extract cdf
					hist = np.histogram(blot.flatten(), num_bins, [0, num_bins], density = False)[0]
					cdf = hist.cumsum()

					## append blots and features to lists
					self.fingerprint["descriptor"].append(des)
					self.fingerprint["blots"].append(rotated_blot)
					self.fingerprint["axes"].append((MA,ma))
					self.fingerprint["cdfs"].append(cdf)


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
			raise ValueError("Must execute blotExtractor before calling function")

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



	def imshowBlots(self, title, draw_axes=False):
		"""
		plots an array of all blots in the image

		Parameters: 
		-------------
		title: str
			String that sets the title for the figure
		draw_axes: boolean (default: False)
			if True, draws principal axes and orientation for each blot

		"""

		if len(self.fingerprint["blots"]) == 0:
			raise ValueError("Must execute blotExtractor before calling function")

		num_blots = len(self.fingerprint["blots"])
		subplot_shape = np.floor(np.sqrt(num_blots)) + 1

		f, axis = plt.subplots(int(subplot_shape),int(subplot_shape))
		f.suptitle(title)
		for i, ax in enumerate(axis.flatten()):
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

			if ax in axis.flatten()[0:num_blots]:
				blot = self.fingerprint["blots"][i].copy()
				
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
			raise ValueError("Must execute blotExtractor before calling function")

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



def main():
	## load in western blot data
	#data = loadWesternBlotData("pathToFile.json")
	data = np.load("testData/test_westernblots.npy") # need to bring in as list of jpeg files

	Fingerprints = []
	for figure_data in data:
		WB = WesternBlot(figure_data)
		WB.westernBlotExtractor()
		WB.blotExtractorWatershed(255)

		Fingerprints.append(WB.fingerprint)

	output_filename = "testData/fingerprintDatabase.npy"
	np.save(output_filename, Fingerprints)



if __name__ == '__main__':
  main()


