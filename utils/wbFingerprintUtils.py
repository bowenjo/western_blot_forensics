import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.isotonic import IsotonicRegression
from scipy import ndimage as ndi 
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmin
from scipy.spatial import KDTree
import itertools as itt
import time
"""
Helpful utility functions for fingerprinting western blot images. 

"""


## ====================================================================================================================================
## =============================================== Histogram of Oriented Gradients ====================================================
## ====================================================================================================================================

def HOG(image, blockShape=(6,24), binShape = (2,4), orientations = 8, L2_NORMALIZE = True, Visualize = False):
	""" 
	performs a simplified histogram of oriented gradients operation on magnitude(mag) and direction(angle) arrays

	Parameters:
	------------
	image: numpy array
		numpy array of grayscale intensity values
	binShape: tuple (default: (2,4))
		determines the number/shape of block bins for which to perform HG
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

	if image.shape[0] < binShape[0] or image.shape[1] < binShape[1]:
		print("image of shape {} must not be less than binShape".format(image.shape))
		return(None)

	# image = cv2.resize(image, blockShape, cv2.INTER_NEAREST)

	## Compute gradient at each resampled pixel
	mag, angle = sobel_grad(image)

	## find gaussian wighting kernel
	(y, x) = mag.shape # find kernel size
	if y % 2 == 0: # make sure y is positive and odd
		y_odd = y + 1  
	else: 
		y_odd = y
	if x % 2 == 0: # make sure x is positive and odd
		x_odd = x + 1 
	else:
		x_odd = x

	Gkernel = _getGaussianKernel2D(y_odd, x_odd, sigma1=y_odd/2, sigma2=x_odd/2, maxValue=1) # get 2D gaussian kernel
	rsmpld_Gkernel = cv2.resize(Gkernel, (x, y)) # resample to fit size of mag

	## weight the magnitudes by gaussian weighting function
	mag_weighted = mag * rsmpld_Gkernel

	## Perform histogram of oriented gradients on magnitude and oreintation arrays
	bin_width = 360. / orientations # set bin width to span full 360 degrees
	feature_vector = np.array([]) # intialize feature vector 
	y_diff = int(np.floor(y / binShape[0])) # step size in vertical direction
	x_diff = int(np.floor(x / binShape[1])) # step size in horizontal direction

	## Loop over row dimension
	row_idx = 0
	for row in range(binShape[0]):
		## Loop over column dimenion
		col_idx = 0
		for col in range(binShape[1]):

			# find the shift index to sum over
			if row == binShape[0] - 1:
				y_shift = y+1
			else: 
				y_shift = row_idx + y_diff
			if col == binShape[1] - 1:
				x_shift = x+1
			else:
				x_shift = col_idx + x_diff

			block_angle = angle[row_idx:y_shift, col_idx:x_shift] # block bins for angles
			block_mag = mag_weighted[row_idx:y_shift, col_idx:x_shift] # block bins for magnitudes
			
			## trilinear interpolation method
			oriented_mags = np.zeros(orientations) # intitalize magnitude accumulation vector 
			centers = np.arange(bin_width / 2, 360 + bin_width / 2, bin_width)  # vector of centers of each bin
			for idx, ang in enumerate(block_angle.flatten()): # loop through each block angle
				d = np.abs(centers - ang) / bin_width # distance from angle value to centers of bin in terms of binwidth 
				w = np.maximum(1-d, 0) # interpolate for distance within one bin width
				oriented_mags += w * block_mag.flatten()[idx] # weight each magnitude by distance within one bin width of centers.

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
		return(image, rsmpld_Gkernel, gx, gy, feature_vector, x_diff, y_diff)
	else:
		return(feature_vector)

def sobel_grad(image, DEGREES = True, ABSOLUTE = False):
	gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = -1) # x:gradient
	gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = -1) # y:gradient

	if ABSOLUTE:
		gx = np.absolute(gx)
		gy = np.absolute(gy)

	mag, theta = cv2.cartToPolar(gx, gy, angleInDegrees=DEGREES)

	return(mag, theta)

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

def dd_dc_ellipse(image, e):
    # right side:
    dd_1_weight = weight_side(image, e, (e[2]-90, e[2]+90))
    # left side
    dd_2_weight = weight_side(image, e, (e[2]+90, e[2]+270))
    
    # top:
    dc_1_weight = weight_side(image, e, (e[2]-180, e[2]))
    # bottom
    dc_2_weight = weight_side(image, e, (e[2], e[2]+180))
    
    print("dd_weight", dd_1_weight - dd_2_weight)
    print("dc_weight", dc_1_weight - dc_2_weight)
    if dd_1_weight >= dd_2_weight:
        if dc_1_weight >= dc_2_weight:
            return (1, 1)
        else:
            return (1, -1)                
    else:
        if dc_1_weight >= dc_2_weight:
            return (-1, -1)
        else:
            return (-1, 1)

def dd_dc_box(image):
    h, w = image.shape
    c = (int(h/2), int(w/2))

    # dd_1_weight = np.sum(image[:, c[1]:] == 217)
    # dd_2_weight = np.sum(image[:, :c[1]] == 217)
    # dc_1_weight = np.sum(image[:c[0], :] ==217)
    # dc_2_weight = np.sum(image[c[0]:, :] == 217)
    dd_1_weight = np.sum(np.float32(image[:, c[1]:]))
    dd_2_weight = np.sum(np.float32(image[:, :c[1]]))
    dc_1_weight = np.sum(np.float32(image[:c[0], :]))
    dc_2_weight = np.sum(np.float32(image[c[0]:, :]))

    # print("right", dd_1_weight)
    # print("left", dd_2_weight)
    # print("up", dc_1_weight)
    # print("down", dc_2_weight)

    if dd_1_weight >= dd_2_weight:
        if dc_1_weight >= dc_2_weight:
            # print("up-right")
            return (1, 1)
        else:
            # print("down-right")
            return (1, -1)                
    else:
        if dc_1_weight >= dc_2_weight:
            # print("up-left")
            return (-1, -1)
        else:
            # print("down-left")
            return (-1, 1)
            
def weight_side(image, e, range_angle):
    background = np.zeros_like(image)
    mask = cv2.ellipse(background, e[0], e[1], e[2], range_angle[0], range_angle[1], (255), -1)
    # plt.figure()
    # plt.imshow(background, "gray")
    # plt.show()
    return np.mean(image[np.where(mask == 255)])

def rotate_box(box, rotation_matrix):
	length = len(box)
	box_h = np.hstack((box,np.ones((length,1)))) # convert to homogeneous
	return box_h @ rotation_matrix.T



def dominant_gradient_direction(theta, mag, num_bins):
	"""
	picks the diminant gradient direction given a gradient orientation and magnitude array

	Parameters:
	------------
	theta: numpy array
		gradient orientation array
	mag: numpy array
		gradient magnitude array
	num_bins: int
		the number of orientation bins (between 0 and 360 degrees)

	Returns:
	---------
	dominant_dir: float
		the dominant orientation of the gradient in degrees
	"""
	bin_width = 360./num_bins
	centers = np.arange(bin_width/2, 360 + bin_width/2, bin_width)

	oriented_mags = np.zeros(num_bins)
	for i, ang in enumerate(theta.flatten()):
		d = np.abs(centers - ang) / bin_width
		w = np.maximum(1-d, 0)
		oriented_mags += w * mag.flatten()[i]
	norm = oriented_mags / np.linalg.norm(oriented_mags)

	return 90 - (bin_width * np.argmax(norm))


def dominant_curl(theta, mag):
	"""
	computes the sign of the dominant curl of a gradient field

	Parameters:
	------------
	theta: numpy array
		gradient orientation array
	mag: numpy array
		gradient magnitude array

	"""
	np.sign(np.sum(mag * theta))













## ====================================================================================================================================
## =============================================== Histogram Equalization =============================================================
## ====================================================================================================================================


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











## ====================================================================================================================================
## =============================================== Figure Image Filtering =============================================================
## ====================================================================================================================================


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
			cnt_f = np.where(areas < (beta * areas[cnt_t]))[0][0] # designate minimum area index
			if cnt_f is None:
				cnt_f = len(areas)
		except IndexError:
			cnt_f = len(areas)

		return([cnt for cnt in contours if cv2.contourArea(cnt) in areas[cnt_t:cnt_f+1]]) # filter contours by designated bounds

def approxAndExtractRect(figure, clf, contours, hierarchy, epsilon, lwr_size_lim, upr_size_lim):
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

	contours_parent = contours_chain(figure, contours, hierarchy, percent_bckgrnd=.25)

	for cnt in contours_parent:
		if cv2.contourArea(cnt) <= (4*lwr_size_lim**2):
			continue
		percArc = epsilon * cv2.arcLength(cnt, True) # accuracy value a percentage of contour arc length
		approx = cv2.approxPolyDP(cnt, percArc, True) # approximate the contour to a simpler shape (i.e diagnol line, rectangle, etc.)
		if len(approx) >= 1:
			x,y,w,h = cv2.boundingRect(approx) # create bounding rectangle to extract region of interest around approximated contour

			# filter for size
			if w<=lwr_size_lim or h<=lwr_size_lim:
				continue
			if w>=upr_size_lim or h>=upr_size_lim:
				continue

			roi = figure[y:np.minimum(y+h, figure_gray.shape[0]), x:np.minimum(x+w, figure_gray.shape[1]), :]
			figures["images"].append(roi)
			figures["grays"].append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
			figures["loc"].append((x,y,w,h))


	if clf is not None:
		clf_idx = []
		for i, image in enumerate(figures["images"]):
			#resized = cv2.resize(image, (100,50), cv2.INTER_LINEAR)
			resized = cv2.resize(image, (200,100), cv2.INTER_LINEAR)
			normlaized = resized / 255
			#reshaped = normlaized.reshape(1, 50, 100, 3)
			reshaped = normlaized.reshape(1, 100, 200, 3)
			if clf.eval(reshaped) == 1:
				clf_idx.append(i)
		figures = _updateDictionary(figures, clf_idx)
		if figures is None:
			return(None)

	return(figures)

def contours_chain(image, contours, hierarchy, percent_bckgrnd):
	"""
	extracts contour chain. If parent contour contains a lot of empty space, will extract corresponding child contours instead.

	INPUTS:
	-----------
	image: numpy array
		the image containing the contours
	contours: list of list
		contour points from image
	hierarchy: numpy array
		opencv contour hierarchy tree
	percent_bckgrnd: float
		the percentage of size allotted to be empty space

	OUTPUTS:
	-----------
	final_cnts: list of list
		parent contours and non-empty space child contours
	"""

	def append_contours(parent_indices, parent_contours, hierarchy, contours, image, percent_bckgrnd, final_cnts):
		"""
		Appends:
		1) parent contours with no children
		2) parent contours with less than percent_bckgrnd empty space, or
		3) children contours with less than percent_bckgrnd empty
		"""
		for cnt_i, cnt in zip(parent_indices, parent_contours):
			if hierarchy[0, cnt_i, 2] == -1:
				final_cnts.append(cnt)
			else:
				x,y,w,h = cv2.boundingRect(cnt)
				area = image[y:y+h, x:x+w]

				if area[np.where(area == 255)].size/area.size > percent_bckgrnd:
					child_indices = np.where(hierarchy[0,:,3] == cnt_i)[0]
					child_cnts = np.array(contours)[child_indices]

					append_contours(child_indices, child_cnts, hierarchy, contours, image, percent_bckgrnd, final_cnts)
				else:
					final_cnts.append(cnt)

	# find parents in hierarchy
	parent_indices = np.where(hierarchy[0,:,3] == -1)[0]
	# extract parent contours
	parent_contours = np.array(contours)[parent_indices]

	final_cnts = []
	append_contours(parent_indices, parent_contours, hierarchy, contours, image, percent_bckgrnd, final_cnts)

	return list(final_cnts)

def _parent_contours(contours, hierarchy):
	## TODO: break apart contours with many large non-overlapping children

	parent_indices = np.where(hierarchy[0,:,3] == -1) # find parents in hierarchy
	parent_contours = contours[parent_indices[0]]

	return list(parent_contours)


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
			try:
				dictionary[key] = np.array(dictionary[key])[index]
			except ValueError:
				return(None)

	return(dictionary)


def blotFilter(image, x, y, w, h, clf, window_stretch=20):
	if clf is not None:
		x1 = np.maximum(x-window_stretch, 0)
		y1 = np.maximum(y-window_stretch, 0)
		x2 = np.minimum(x1 + w + 2*window_stretch, image.shape[1])
		y2 = np.minimum(y1 + h + 2*window_stretch, image.shape[0])

		color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		blot_region = color[y1+1:y2, x1+1:x2]
		resized = cv2.resize(blot_region, (30,15), cv2.INTER_LINEAR)
		normalized = resized/255
		reshaped = normalized.reshape(1,15,30,3)

		return clf.eval(reshaped)
	else:
		return 1







## ====================================================================================================================================
## =============================================== Segmentation =======================================================================
## ====================================================================================================================================
def get_rect(x,y,w,h,shape, ws=0):
	p1 = np.maximum(0, y - ws)
	p2 = np.maximum(0, x - ws)
	p3 = np.minimum(y + h + ws, shape[0])
	p4 = np.minimum(x + w + ws, shape[1])

	return p1, p2, p3, p4

def get_blot_points(image_locs, blot_locs):
	x = image_locs[0] + blot_locs[0]
	w = blot_locs[2]
	y = image_locs[1] + blot_locs[1]
	h = blot_locs[3]

	return(x,y,w,h)

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
		min_loc = cv2.minMaxLoc(image, mask=mask)[3]
		cv2.circle(fine_markers, min_loc, 1, (255), -1)

	fine_markers = cv2.connectedComponents(fine_markers)[1]

	return(fine_markers)

def dtwGradient(image, binary_image, alpha):
	DT = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
	DT_max = np.max(DT)
	DT_min = np.min(DT)

	dtwImage = image * np.exp(-alpha * (DT-DT_min)/(DT_max - DT_min))

	return(cv2.cvtColor(dtwImage, cv2.COLOR_GRAY2BGR).astype(np.uint8), DT)









## ====================================================================================================================================
## =============================================== Over-segmentation Merging ==========================================================
## ====================================================================================================================================


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









## ====================================================================================================================================
## =============================================== Contour Properties =================================================================
## ====================================================================================================================================

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
	dist = np.max(dist) - dist
	mu_dt, stdDev_dt = cv2.meanStdDev(dist, mask=mask)
	circularity = (mu_dt/(stdDev_dt + np.finfo(float).eps))

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




## ====================================================================================================================================
## =============================================== Logging and Time ===================================================================
## ====================================================================================================================================

def print_time(sec):
	t = time.localtime(sec)
	time_str = "{}/{}/{} {}:{}:{}".format(t.tm_mon, t.tm_mday, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec)
	return time_str




































































