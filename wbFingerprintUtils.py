import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmax
import itertools
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
	equ_image = list(map(lambda x: equalizer(x, equ_LookUp), list(image.flatten())))
	equ_image = np.array(equ_image).reshape(image.shape)

	return(equ_image)



def equalizer(v, equ_LookUp):
	return(equ_LookUp[v])


def normalize(array):
	return((array-np.min(array))/(np.max(array) - np.min(array) + np.finfo(float).eps))


def distThresh(dist_transform, spacing_num):
	"""
	computes the distance threshold based on a weighting function maximizing number of contours and total area.
	
	#TODO: update weighting method for increased accuracy at extracting blots. Perhaps ncorporate variance and mean area. 

	Parameters:
	------------
	dist_transform: numpy array
		array of values describing the distance threshold of the contours
	spacing_num: int
		defines the number of values to try between zero and one
	area_weight: float (default: .35)
		weights the effect total area has on determination of the distance threshold

	"""
	stats = {"num_cnts":[]}

	possible_thresh = np.linspace(0, 1, spacing_num)[1:]
	for t in possible_thresh:
		thresh_image = np.uint8(cv2.threshold(dist_transform,t*dist_transform.max(),255,0)[1])
		cnts = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
		stats["num_cnts"].append(len(cnts))

	weight = (stats["num_cnts"])

	return(possible_thresh[np.argmax(weight)]) 

def adaptiveDilation(binary_image, maxKernelSize=5, iterations=1, structure = cv2.MORPH_ELLIPSE):
	"""
	Does adaptive morphological dilation on binary images. Designed to do more dilation for segments with smaller area, 
	and less dilation for those with large area. 

	#TODO: use the variance (or other statistical measure) to weight the kernel size. So when the variance of the blots are lower, do less dilation. 

	Parameters:
	-------------
	binary_image: numpy array
		binary image to have dilation applied to
	maxKernelSize: int (default = 7)
		sets the maximum kernel size for dilation
	iterations: int (default = 1)
		sets the number of iterations to perform morphological dilation
	structure: cv2 object (default = cv2.MORPH_ELLIPSE)
		determines the shape of the kernel used for dilation

	Returns:
	------------
	dilated_image: numpy array
		output binary image after adaptive dilation
	"""
	copy = binary_image.copy()

	contours = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1] # find outlines of binary image
	areas = np.array([cv2.contourArea(cnt) for cnt in contours]) # compute area
	kernelSizes = np.around(maxKernelSize * (1 - areas / np.sum(areas))).astype(int) # get the kernel size weighted by area

	dilated_image = np.zeros(binary_image.shape, dtype=np.uint8) # intitialize dileated image
	for i, cnt in enumerate(contours):
		mask = np.zeros(binary_image.shape, dtype=np.uint8) # intitialize background mask
		cv2.drawContours(mask, [cnt], 0, (255), -1) # fill in contour outline for each contour

		if kernelSizes[i] % 2 == 0: # make sure kernel size is odd
			kernelSizes[i] = kernelSizes[i] + 1

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSizes[i], kernelSizes[i])) # create kernel
		mask = cv2.dilate(mask, kernel, iterations = iterations) # dilate masked image for each filled in contour
		dilated_image = cv2.add(dilated_image, mask) # add new filled-in outlined image to dilated image 
	
	return(dilated_image)




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

	Gkernel = getGaussianKernel2D(ksize1_odd, ksize2_odd, sigma1=ksize1_odd/2, sigma2=ksize2_odd/2, maxValue=1) # get 2D gaussian kernel
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
		feature_vector = feature_vector/norm

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

def getGaussianKernel2D(ksize1, ksize2, sigma1, sigma2, maxValue=1):
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
		areas = np.sort([cv2.contourArea(cnt) for cnt in contours])[::-1] # all areas
		cnt_t = np.where(areas < (figure_size - alpha))[0][0] # designated maximum area index
		cnt_f = np.where(areas < beta*areas[cnt_t])[0][0] # designate minimum area index

		return([cnt for cnt in contours if cv2.contourArea(cnt) in areas[cnt_t:cnt_f+1]]) # filter contours by designated bounds

def approxAndExtractRect(figure, contours, epsilon, size_lim, whr_lim, mode_lim, shift_length, hist_lim):
	# initialize vectors
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
	masks = [rectMask(figure_gray.shape, loc) for loc in figures["loc"]]
	mask_idx = list(itertools.combinations(range(len(masks)), 2))
	masks_combos = itertools.combinations(masks, 2)

	delete_idx = []
	for i, combo in enumerate(masks_combos):
		combined = np.sum(np.add(combo[0], combo[1])) # look at mask intercept
		if np.any(combined==2):
			if np.size(combo[0]) < np.size(combo[1]):
				delete_idx.append(mask_idx[i][0])
			else:
				delete_idx.append(mask_idx[i][1])

	overlap_idx = [i for i, loc in enumerate(figures["loc"]) if i not in delete_idx]
	figures = updateDictionary(figures, overlap_idx)

	print(overlap_idx)

	# filter by size
	sizes = np.sort([np.size(roi) for roi in figures["grays"]])[::-1]
	size_t, size_f = filterByIndex(sizes, size_lim)
	size_idx = [i for i, roi in enumerate(figures["grays"]) if np.size(roi) in sizes[size_t:size_f]]
	figures = updateDictionary(figures, size_idx)

	print(size_idx)

	# filter by shape ratio
	wh_ratio = np.sort([roi.shape[1]/roi.shape[0] for roi in figures["grays"]])[::-1]
	wh_ratio_t, wh_ratio_f = filterByIndex(wh_ratio, whr_lim)
	shape_idx = [i for i, roi in enumerate(figures["grays"]) if (roi.shape[1]/roi.shape[0]) in wh_ratio[wh_ratio_t:wh_ratio_f]]
	figures = updateDictionary(figures, shape_idx)

	print(shape_idx)

	# filter based on mode of histograms
	hists = [np.histogram(roi.flatten(), np.arange(256), density=False)[0] for roi in figures["grays"]]
	mode_ratio = np.sort([np.max(hist) for hist in hists])[::-1]
	mode_ratio_t, mode_ratio_f = filterByIndex(mode_ratio, mode_lim)
	mode_idx = [i for i, hist in enumerate(hists) if np.max(hist) in mode_ratio[mode_ratio_t:mode_ratio_f]]
	figures = updateDictionary(figures, mode_idx) 

	print(mode_idx)

	# filter based on distribution around mode
	hists = [np.histogram(roi.flatten(), np.arange(256), density=False)[0] for roi in figures["grays"]]
	hist_idx = []
	for i, hist in enumerate(hists):
		mode = np.argmax(hist)
		try:
			# right-shifted ratio
			right_shift = np.arange(1, shift_length+1) + mode
			if hist[mode] / np.sum(hist[right_shift]) < hist_lim:
				hist_idx.append(i)
		except IndexError:
			# left-shifted ratio
			left_shift = np.arange(-shift_length-1, -1) + mode
			if hist[mode] / np.sum(hist[left_shift]) < hist_lim:
				hist_idx.append(i)
		except RuntimeWarning:
			continue
	print(hist_idx)

	figures = updateDictionary(figures, hist_idx)
	#del figures["grays"]

	return(figures)

def filterByIndex(sorted_list, limit):
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

def rectMask(shape, rect):
	mask = np.zeros(shape)
	mask[rect[1]:np.minimum(rect[1]+rect[3], shape[0]), rect[0]:np.minimum(rect[0]+rect[2], shape[1])] = 1
	return(mask)

def updateDictionary(dictionary, index):
	if len(index) == 0:
		print("Had no effect on updating the dictionary")
		pass
	else:
		for key in dictionary.keys():
			dictionary[key] = np.array(dictionary[key])[index]

	return(dictionary)

def contrastFilters(figure, break_point):
	"""
	Creates a ReLU function with breaking point defined by break_point parameter.
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








