import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.isotonic import IsotonicRegression
from scipy import ndimage as ndi 
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmin
from scipy.spatial import KDTree
import itertools as itt
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.stats import multivariate_normal
"""
Helpful utility functions for fingerprinting western blot images. 

"""


def equalizeImage(image, num_bins = 256):
	"""
	Equalizes the histogram of a single image and returns the equalized image and cdf.

	TODO: put in error for non-gray-scale image, error for non-uint8

	Parameters:
	------------
	image: numpy array (dtype = uint8)
		Grayscale image 
	num_bins: int (default = 256)
		number of bins to break up the histogram.

	Returns:
	------------
	equ_image: numpy array (stype = uint8)
		equalized image
	equ_cdf: numpy array
		array of values of the cummulative distribution function of the equalized image

	"""
	image_noBlack = [x for x in image.flatten() if x!=0]
	image_noBlack = sorted(list(set(image_noBlack)))

	## create the pdf and cdf of the image
	hist, bins = np.histogram(image_noBlack, num_bins, [0, num_bins], density = False)
	cdf = hist.cumsum()
	cdf = sorted(list(set([x for x in cdf if x!=0])))
	cdf_min = np.min(cdf) #minimum non-zero value of cdf

	equ = np.round((num_bins-1)*np.array((cdf)/(cdf[-1]))) # general equalization function

	## create look-up table mapping pixel value to equalized pixel value
	equ_LookUp = dict(zip(image_noBlack,equ)) 
	equ_LookUp[0] = 0 # So background is mapped to zero


	## equalize image
	equ_image = list(map(lambda x: _equalizer(x, equ_LookUp), list(image.flatten())))
	equ_image = np.array(equ_image).reshape(image.shape)

	return(equ_image)


def _equalizer(v, equ_LookUp):
	return(equ_LookUp[v])


def HOG(image, blockShape = (4,8), binShape = (2,4), orientations = 8, L2_NORMALIZE = True, Visualize = False):
	""" 
	performs a simplified histogram of oriented gradients operation on magnitude(mag) and direction(angle) arrays

	# TODO: put in Gaussian bias function to weight gradient values closer to centroid higher than those in the periphery 

	Parameters:
	------------
	image: numpy array
		numpy array of grayscale intensity values
	blockShape: tuple (default: (4,8))
		determines the resampling size of downsampled array
	binShape: tuple (default: (2,4))
		determines the number/shape of block bins for which to perform HOG
	orientations: int (default: 8)
		determines the number of orientation bins
	L2_NORMALIZE: boolean (default: True)
		determines if feature vector should be L2-noramlized
	Visualize: boolean (default: False)
		if true will output visualization plots

	Returns:
	-----------
	feature_vector: numpy array (length = orientations)
		accumulated magnitudes of specified orientation bin
	"""
	if blockShape[0] % binShape[0] != 0 or blockShape[1] % binShape[1] != 0:
		raise ValueError("blockShape must be a whole-number multiple of binShape")

	## Resample image to blockShape
	rsmpld_image = cv2.resize(image, (blockShape[1], blockShape[0]), interpolation = cv2.INTER_NEAREST)
	#rsmpld_image = cv2.GaussianBlur(rsmpld_image, (3,3), 1)

	## Compute gradient at each resampled pixel
	gx = cv2.Sobel(rsmpld_image, cv2.CV_64F, 1, 0, ksize = 1) # x:gradient
	gy = cv2.Sobel(rsmpld_image, cv2.CV_64F, 0, 1, ksize = 1) # y:gradient

	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True) # convert gradient in each cardinal direction to magnitude and angle

	## find gaussian wighting kernel
	(ksize1, ksize2) = mag.shape # find kernel size
	if ksize1 % 2 == 0: # make sure ksize1 is positive and odd
		ksize1_odd = ksize1 + 1  
	else: 
		ksize1_odd = ksize1
	if ksize2 % 2 == 0: # make sure ksize2 is positive and odd
		ksize2_odd = ksize2 + 1 
	else:
		ksize2_odd = ksize2

	Gkernel = _getGaussianKernel2D(ksize1_odd, ksize2_odd, sigma1=ksize1_odd/2, sigma2=ksize2_odd/2, maxValue=1) # get 2D gaussian kernel
	rsmpld_Gkernel = cv2.resize(Gkernel, (ksize2, ksize1)) # resample to fit size of mag

	## weight the magnitudes by gaussian weighting function
	mag_weighted = mag * rsmpld_Gkernel

	## Perform histogram of oriented gradients on magnitude and oreintation arrays
	bin_width = 360. / orientations # set bin width to span full 360 degrees
	feature_vector = np.array([]) # intialize feature vector 
	y_diff = int(blockShape[0] / binShape[0]) # step size in vertical direction
	x_diff = int(blockShape[1] / binShape[1]) # step size in horizontal direction

	## Loop over row dimension
	row_idx = 0
	for row in range(binShape[0]):
		## Loop over column dimenion
		col_idx = 0
		for col in range(binShape[1]):
			block_angle = angle[row_idx:row_idx+y_diff, col_idx:col_idx+x_diff] # block bins for angles
			block_mag = mag_weighted[row_idx:row_idx+y_diff, col_idx:col_idx+x_diff] # block bins for magnitudes
			
			## trilinear interpolation method
			oriented_mags = np.zeros(orientations) # intitalize magnitude accumulation vector 
			centers = np.arange(bin_width / 2, 360 + bin_width / 2, bin_width)  # vector of centers of each bin
			for idx, ang in enumerate(block_angle.flatten()): # loop through each block angle
				d = np.abs(centers - ang) / bin_width # distance from angle value to centers of bin in terms of binwidth 
				w = np.maximum(1-d, 0) # interpolate for distance within one bin width
				oriented_mags += w * block_mag.flatten()[idx] # weight each magnitude by distance within one bin width of centers.

			## No interpolation method
			# bins = [np.where(block_angle<=((idx+1)*bin_width)) for idx in range(orientations)] # label where orientations in angles is within a designated bin
			# oriented_mags = [np.sum(block_mag[b]) for b in bins] # accumulate magnitudes in given orientation bin
			# oriented_mags = np.hstack((np.array([oriented_mags[0]]), np.diff(oriented_mags))) # make sure non-cummulative between bins

			# if L2_NORMALIZE is True: # l2 normalize over each block
			# 	norm = np.linalg.norm(oriented_mags)
			# 	oriented_mags = oriented_mags/norm

			feature_vector = np.append(feature_vector, np.array(oriented_mags))

			col_idx += x_diff

		row_idx += y_diff

	if L2_NORMALIZE is True: # l2 normalize over each bin
		norm = np.linalg.norm(feature_vector)
		if norm != 0:
			feature_vector = feature_vector/norm
		else:
			return(None)

	if Visualize is True:
		return(rsmpld_image, rsmpld_Gkernel, gx, gy, feature_vector)
	else:
		return(feature_vector)


def pickBlockSize(ma, MA, blockShape=(4,8)):
	"""
	chooses a pixel-wise block size based off of major and minor axes of ellipse. It is ideal to have a block size that is some
	whole-number multiple of of the sampled-down cell shape. This will be used in a histogram-of-oriented-gradient-inspired method 
	to extract a decriptor vector from individual interest points.  

	Parameters:
	-------------
	ma: float
		minor axis of an orientation-normalized ellipse
	MA: float
		major axis of and orientation-normalized ellipse
	cellshape: tuple (default: (4,8))
		defines the sampling size for the array

	Returns:
	-------------
	(y, x): floats
		the vertical and horizontal dimensions of the pixel-wise block
	"""
	## Compute vertical component of pixel-wise block
	y = int((np.floor(ma / blockShape[0]) + 1) * blockShape[0])
	## Compute horizontal component of pixel-wise block
	x = int((np.floor(MA / blockShape[1]) + 1) * blockShape[1])

	return(y,x)

def _getGaussianKernel2D(ksize1, ksize2, sigma1, sigma2, maxValue=1):
	"""
	creates a a 2D gaussian kernel of dimension (ksize1 x ksize2) with standard deviation sigma and maximum value equal to maxValue

	Parameters:
	------------
	ksize1: int (odd and positive)
		row dimension of kernel
	ksize2: int (odd and positive)
		column dimension of kernel
	sigma1: float
		standard deviation of row-dimension kernel
	simga2: float
		standard deviation of column-dimension kernel
	maxValue: float
		maximum value of gaussian function 

	Returns:
	------------
	kernel: numpy array
		2D gaussian kernel with dimensions (ksize1, ksize2) 
	"""

	kern1 = cv2.getGaussianKernel(ksize1, sigma1) # 1st 1D kernel
	kern2 = cv2.getGaussianKernel(ksize2, sigma2) # 2nd 1D kernel
	kernel = np.sqrt(np.outer(kern1, kern2)) # compute squareroot of outer product to get 2D kernel

	kernel = (maxValue / np.max(kernel)) * kernel # scale to get maximum value = maxValue

	return(kernel)

def filterContourArea(contours, figure_size, alpha=1000, beta=.01):
		if len(contours) == 0:
			return(None)

		areas = np.sort([cv2.contourArea(cnt) for cnt in contours])[::-1] # all areas
		try:
			cnt_t = np.where(areas < (figure_size - alpha))[0][0] # designated maximum area index
			if cnt_t is None:
				cnt_t = 0
		except IndexError:
			cnt_t = 0
		try:
			cnt_f = np.where(areas < beta*areas[cnt_t])[0][0] # designate minimum area index
			if cnt_f is None:
				cnt_f = len(areas)
		except IndexError:
			cnt_f = len(areas)

		return([cnt for cnt in contours if cv2.contourArea(cnt) in areas[cnt_t:cnt_f+1]]) # filter contours by designated bounds

def approxAndExtractRect(figure, contours, epsilon, size_lim, whr_lim, mode_lim, shift_length, hist_lim):
	# initialize vectors
	if contours is None:
		print("There were no contours in the image")
		return(None)

	figures = {"images": [],
			   "grays": [],
			   "loc": []
			   }

	# find contours in the figure
	figure_gray = cv2.cvtColor(figure, cv2.COLOR_BGR2GRAY)

	for cnt in contours:
		percArc = epsilon * cv2.arcLength(cnt, True) # accuracy value a percentage of contour arc length
		approx = cv2.approxPolyDP(cnt, percArc, True) # approximate the contour to a simpler shape (i.e diagnol line, rectangle, etc.)
		x,y,w,h = cv2.boundingRect(approx) # create bounding rectangle to extract region of interest around approximated contour
		roi = figure[y:np.minimum(y+h, figure_gray.shape[0]), x:np.minimum(x+w, figure_gray.shape[1]), :]
		figures["images"].append(roi)
		figures["grays"].append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
		figures["loc"].append((x,y,w,h))

	# delete overlapping regions
	masks = [_rectMask(figure_gray.shape, loc) for loc in figures["loc"]]
	mask_idx = list(itt.combinations(range(len(masks)), 2))
	masks_combos = list(itt.combinations(masks, 2))

	delete_idx = []
	for i, combo in enumerate(masks_combos):
		combined = np.add(combo[0], combo[1])# look at mask intercept
		if np.any(combined==2):
			if np.size(combo[0]) < np.size(combo[1]):
				delete_idx.append(mask_idx[i][0])
			else:
				delete_idx.append(mask_idx[i][1])

	overlap_idx = [i for i, loc in enumerate(figures["loc"]) if i not in delete_idx]
	figures = _updateDictionary(figures, overlap_idx)
	if figures is None:
		return(None)


	# # filter by size
	# sizes = np.sort([np.size(roi) for roi in figures["grays"]])[::-1]
	# size_t, size_f = _filterByIndex(sizes, size_lim)
	# size_idx = [i for i, roi in enumerate(figures["grays"]) if np.size(roi) in sizes[size_t:size_f]]
	# figures = _updateDictionary(figures, size_idx)
	# if figures is None:
	# 	return(None)

	# # filter by shape ratio
	# wh_ratio = np.sort([roi.shape[1]/roi.shape[0] for roi in figures["grays"]])[::-1]
	# wh_ratio_t, wh_ratio_f = _filterByIndex(wh_ratio, whr_lim)
	# shape_idx = [i for i, roi in enumerate(figures["grays"]) if (roi.shape[1]/roi.shape[0]) in wh_ratio[wh_ratio_t:wh_ratio_f]]
	# figures = _updateDictionary(figures, shape_idx)
	# if figures is None:
	# 	return(None)

	# # filter based on mode of histograms
	# hists = [np.histogram(roi.flatten(), np.arange(256), density=False)[0] for roi in figures["grays"]]
	# mode_ratio = np.sort([np.max(hist) for hist in hists])[::-1]
	# mode_ratio_t, mode_ratio_f = _filterByIndex(mode_ratio, mode_lim)
	# mode_idx = [i for i, hist in enumerate(hists) if np.max(hist) in mode_ratio[mode_ratio_t:mode_ratio_f]]
	# figures = _updateDictionary(figures, mode_idx) 
	# if figures is None:
	# 	return(None)

	# # filter based on distribution around mode
	# hists = [np.histogram(roi.flatten(), np.arange(256), density=False)[0] for roi in figures["grays"]]
	# hist_idx = []
	# for i, hist in enumerate(hists):
	# 	mode = np.argmax(hist)
	# 	try:
	# 		# right-shifted ratio
	# 		right_shift = np.arange(1, shift_length+1) + mode
	# 		if hist[mode] / (np.sum(hist[right_shift]) + np.finfo(float).eps) < hist_lim:
	# 			hist_idx.append(i)
	# 	except IndexError:
	# 		# left-shifted ratio
	# 		left_shift = np.arange(-shift_length-1, -1) + mode
	# 		if hist[mode] / (np.sum(hist[left_shift]) + np.finfo(float).eps) < hist_lim:
	# 			hist_idx.append(i)

	# figures = _updateDictionary(figures, hist_idx)
	# if figures is None:
	# 	return(None)

	# convert to array
	figures["loc"] = np.array(figures["loc"])
	#del figures["grays"]

	return(figures)

def _filterByIndex(sorted_list, limit):
	try:
		t = np.where(sorted_list < (np.mean(sorted_list) + limit[0] * np.std(sorted_list)))[0][0]
		if t is None:
			t = 0
	except IndexError:
		t = 0
	try:
		f = np.where(sorted_list < (np.mean(sorted_list) - limit[1] * np.std(sorted_list)))[0][0]
		if f is None:
			f = len(sorted_list)
	except IndexError:
		f = len(sorted_list)

	return(t, f)

def _rectMask(shape, rect):
	mask = np.zeros(shape)
	mask[rect[1]:np.minimum(rect[1]+rect[3], shape[0]), rect[0]:np.minimum(rect[0]+rect[2], shape[1])] = 1
	return(mask)

def _updateDictionary(dictionary, index):
	if len(index) == 0:
		return(None)
	else:
		for key in dictionary.keys():
			dictionary[key] = np.array(dictionary[key])[index]

	return(dictionary)

def contrastFilters(figure, break_point):
	"""
	Creates a saturation function with breaking point defined by break_point parameter.
	"""

	if break_point < 0 or break_point > 1:
		raise ValueError("break_point must be a real number between zero and one")

	ydata = np.array([0, 0, 1]) * 255
	xdata = np.array([0, break_point, 1]) * 255

	shape = figure.shape
	
	ir = IsotonicRegression()
	ir.fit(xdata, ydata)
	y_ = ir.transform(figure.flatten()).reshape(shape)

	return(np.uint8(y_))

def CLAHE(image, clipLimit, tileGridSize):
	"""
	Parameters:
	------------
	image: numpy array
	clipLimit: float
	tileGridSize: tuple 

	Returns:
	------------
	adaptive histogram equalizated image

	"""

	clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
	return(clahe.apply(image))


def mergeWatershedLines(locs, image_shape, alpha):
	## GOAL:
	# - assume watershed segemntation is not undersegmented.
	# - find maximum blot dimensions in the image. This assumes that in most cases at least one blot will be segmented correctly
	# - compare other segmentation dimensions to max dimensions.
	# - if segments dimensions << max dimensions and connected to another << max dimensions segment, merge the segments.
	# - consider more than two segments merging. This will account for touching oversegmentations. 

	## TODO: 
	# - for each small-width blot, find closest neighboring blot. If distance between closest neighbor is 
	#	significantly smaller than mean closest-neighbor-distance, merge blots. This will account for non-touching oversegmentations
	# - weight merging from width by multivariate confidence score ratio (post-merge confidence/pre-merge confidence). This will account for 
	#	over-merging due to possible undersegmentation in the image (which is against the assumption made in the goals). 

	# initialize widths
	widths = [rect[2] for rect in locs["loc"]]
	max_w = np.max(widths) 

	# find partial segments form max width calculation
	ps_idx = [idx for idx, w in enumerate(widths) if w <= alpha * max_w]
	fs_idx = [idx for idx, w in enumerate(widths) if w > alpha * max_w]
	ps_cnts = locs["contours"][ps_idx]
	fs_cnts = locs["contours"][fs_idx]

	# draw partial segments
	mask = np.zeros(image_shape, dtype=np.uint8) # create backgroud to superimpose contours on
	cv2.drawContours(mask, ps_cnts, -1, (255), -1) # superimpose contours onto background
	merged = cv2.connectedComponents(mask)[1] # merge connected components 

	# add on full segments
	for i, seg in enumerate(fs_cnts):
		color = int(i + (np.max(merged)+1))
		cv2.drawContours(merged, [seg], 0, (color), -1)

	return(merged)

def minDistCnt(cnt1, cnt2):
	minimum_distances = []
	for pt1 in cnt1:
		min_dist = np.min([np.linalg.norm(pt1 - pt2) for pt2 in cnt2])
		minimum_distances.append(min_dist)

	return(np.min(minimum_distances))

def groupRepeats(N):
	""" 
	groups repeated sequences of a float in a 1-D array
	"""
	pointer = np.append(np.where(np.diff(N)!=0), len(N)-1) # indices pointing to grouped array
	return((pointer, N[pointer]))

def scanForLocalMin(image, T, order, line_type):
	"""
	scans image for local minima with horizontal and vertical scanning-lines 
	"""
	local_min = {'x':[],'y':[]} # initialize dots dictionary
	for idx, line in enumerate(image):
		g_line = groupRepeats(line) # group repeated sequences
		x = g_line[0][argrelmin(g_line[1], order=order)] # find x-components of local min
		y = [idx]*len(x) # find corresponding y-components of local min
		local_min["x"] = np.append(local_min["x"], x) 
		local_min["y"] = np.append(local_min["y"], y)   

	# filter out local min above intensity T
	thresh_idx = [i for i, m in enumerate(np.int64(list(zip(local_min["y"],local_min["x"])))) if image[tuple(m)] <= T]

	if len(thresh_idx) != 0:
		for key in local_min.keys():
			local_min[key] = np.array(local_min[key])[thresh_idx]
	else:
		local_min = None

	return(local_min)

def gridLocalMin(image, T, order):
	"""
	finds the indices for x,y-grid local min
	"""
	horizontal = scanForLocalMin(image, T, order[0], line_type='h')
	vertical = scanForLocalMin(image.T, T, order[1], line_type='v')
	if horizontal is not None:
		h_coord = (np.array(horizontal["y"]).astype(np.int64), np.array(horizontal["x"]).astype(np.int64))
	else:
		h_coord = None
	if vertical is not None:
		v_coord = (np.array(vertical["x"]).astype(np.int64), np.array(vertical["y"]).astype(np.int64))
	else:
		v_coord = None

	return(h_coord, v_coord)

def gridLineDetector(image, pts, vote_lim, dist_thresh, line_type, kernel_size, alpha, mask=None):
	"""
	Draws division lines in the image defined by local minima

	Parameters:
	--------------
	image: numpy array
		image array to draw divide lines on
	pts: tuple of arrays
		x, y coordinates of local minima 
	vote_lim: int
		the number of votes that a divide line must exceed to be drawn
	dist_thresh: int
		the minimum number of pixels between divide lines
	line_type: str
		'h' - horizontal 
		'v' - vertical
	kernel_size: int (odd) 
		size of kernel to sum points over
	alpha: float
		number of standard diviations away from mean to count divide line
	mask: numpy array (same dim as image)
		optional image mask

	Returns:
	-------------
	divide_line_image: numpy array (same dim as image)
		image with divide lines drawn

	"""

	# Draw local minima
	dots = np.zeros_like(image, dtype=np.uint8)
	if pts is None:
		return(image)
	dots[pts] = 1
	# mask the local minima
	if mask is not None:
		dots = cv2.bitwise_and(dots, dots, mask=mask)
		dots = np.float32(dots)

	# sum over the respected axis
	if line_type == 'h':
		votes = np.sum(dots, axis=0)
	elif line_type == 'v':
		votes = np.sum(dots, axis=1)
	else:
		raise NameError(line_type + " is not a recognized line_type")

	# calculate votes with a filter
	kernel = np.ones(kernel_size)
	votes = np.convolve(votes, kernel)

	if np.all(votes==0):
		return(image)

	# Decide on divide lines from alpha parameter
	mean = np.mean(votes[np.where(votes>0)])
	std = np.std(votes[np.where(votes>0)])


	truth_value = (votes > (mean + alpha*std)) * (votes > vote_lim)
	divides = np.where(truth_value)[0]

	# draw divide lines
	last_index = 0
	divide_line_image = image.copy()
	for idx, axis in enumerate(divides):
		if idx > 0:
			if (axis - divides[last_index]) > dist_thresh: # divide lines must be at least dist_thresh pixels between each other
				last_index = idx
				try:
					if line_type == 'h':
						divide_line_image[:,axis]=0
					else:
						divide_line_image[axis,:]=0
				except IndexError:
					continue
		else:
			last_index = idx
			try:
				if line_type == 'h':
					divide_line_image[:,axis]=0
				else:
					divide_line_image[axis,:]=0
			except IndexError:
				continue

	return(divide_line_image)

def findMarkers(image, segmented_image):
	"""
	finds the markers for watershed algorithm given segmented, course binary image
	"""
	coarse_markers = cv2.connectedComponents(segmented_image)[1]
	fine_markers = np.zeros_like(image, dtype=np.uint8)
	for cm in np.unique(coarse_markers)[1:]:
		mask = np.zeros_like(segmented_image, dtype=np.uint8)
		mask[coarse_markers == cm] = 255
		min_loc = cv2.minMaxLoc(image, mask=mask)[2]
		cv2.circle(fine_markers, min_loc, 1, (255), -1)

	fine_markers = cv2.connectedComponents(fine_markers)[1]

	return(fine_markers)

def solidity(contour):
	"""
	finds solidity (ratio of contour area to its convex hull area)
	"""
	cnt_area = cv2.contourArea(contour)
	hull = cv2.convexHull(contour)
	hull_area = cv2.contourArea(hull)

	if hull_area != 0:
		return(cnt_area / hull_area + np.finfo(float).eps)
	else:
		return(0)

def eccentricity(contour):
	if len(contour) > 5:
		(cx, cy), (MA,ma), theta = cv2.fitEllipse(contour)
		eccentricity = MA/(ma + np.finfo(float).eps)
	else:
		eccentricity = 0

	return(eccentricity)

def circularity(contour, image):
	mask = np.zeros(image.shape, dtype=np.uint8)
	cv2.drawContours(mask, [contour], 0, (255), -1)

	dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
	mu_dt, stdDev_dt = cv2.meanStdDev(dist, mask=mask)
	circularity = mu_dt/(stdDev_dt + np.finfo(float).eps)

	return(circularity)

def spread(contour, image, num_bins):
	# hist, bins = np.histogram(image.flatten(), num_bins, [0, num_bins], density = False)
	# spread = np.nonzero(hist)[0].size / (num_bins-1)
	mask = np.zeros(image.shape, dtype=np.uint8)
	cv2.drawContours(mask, [contour], 0, (255), -1)

	hist = cv2.calcHist([image], [0], mask, [num_bins], [0,num_bins])
	spread = np.nonzero(hist)[0].size / (num_bins-1)

	return(spread)

def width(contour):
	width = cv2.boundingRect(cnt)[2]

def blot_stats(contour, image, num_bins):
	"""
	Note: These stats where designed to be used to produce a model for "ideal" blot shape. It was going to be used for merging oversegmented blot.
	However, the variability of the dataset in these statistics makes it difficult to  
	"""
	# eccentricity
	ecc = eccentricity(contour)
	# convexity
	conv = solidity(contour)
	# circularity
	circ  = circularity(contour, image)
	# spread
	spr = spread(contour, image, 256)

	return([ecc, conv, circ, spr])

def dtwGradient(image, binary_image, alpha):
    DT = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    DT_max = np.max(DT)
    DT_min = np.min(DT)
    
    dtwImage = image * np.exp(-alpha * (DT-DT_min)/(DT_max - DT_min))
    
    return(dtwImage.astype(np.uint8), cv2.cvtColor(dtwImage, cv2.COLOR_GRAY2BGR).astype(np.uint8))






























































