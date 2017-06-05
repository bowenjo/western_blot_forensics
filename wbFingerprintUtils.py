import numpy as np 
import cv2
import matplotlib.pyplot as plt 

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


def distThresh(dist_transform, spacing_num, area_weight = .35):
	"""
	computes the distance threshold based on a weighting function maximizing number of contours and total area.

	Parameters:
	------------
	dist_transform: numpy array
		array of values describing the distance threshold of the contours
	spacing_num: int
		defines the number of values to try between zero and one
	area_weight: float (default: .35)
		weights the effect total area has on determination of the distance threshold

	"""
	stats = {"num_cnts":[], 
			 "mean_areas":[], 
			 "var_areas":[]
			}

	possible_thresh = np.linspace(0, 1, spacing_num)[1:]
	for t in possible_thresh:
		thresh_image = cv2.threshold(dist_transform,t*dist_transform.max(),255,0)[1]
		thresh_image = np.uint8(thresh_image)
		cnts = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

		## filter ontours by area
		# areas = [cv2.contourArea(cnt) for cnt in cnts] # list of all contour areas using 0th image moment
		# areas = [a for a in areas if np.abs(a-np.mean(areas)) < np.std(areas)] # remove outliers
		# cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) in areas]

		if len(cnts) != 0:
			mean_area = np.mean([cv2.contourArea(cnt) for cnt in cnts])
			var_area = np.var([cv2.contourArea(cnt) for cnt in cnts])
		else:
			mean_area = 0

		stats["mean_areas"].append(mean_area)
		stats["var_areas"].append(var_area)
		stats["num_cnts"].append(len(cnts))
	# print("Number of blots for each distance threshold value:", num_cnts)	

	##TODO: update weighting method for increased accuracy at extracting blots
	#loc = np.where(np.array(stats["num_cnts"]) == 1)

	stats["num_cnts"] = normalize(np.array(stats["num_cnts"]))
	stats["mean_areas"] = normalize(np.array(stats["mean_areas"]))
	stats["var_areas"] = normalize(np.array(stats["var_areas"]))


	weight = (stats["num_cnts"] + area_weight*stats["mean_areas"])

	return(possible_thresh[0]) #return(possible_thresh[np.argmax(weight)])


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

			bins = [np.where(block_angle<=((idx+1)*bin_width)) for idx in range(orientations)] # label where orientations in angles is within a designated bin
			oriented_mags = [np.sum(block_mag[b]) for b in bins] # accumulate magnitudes in given orientation bin
			oriented_mags = np.hstack((np.array([oriented_mags[0]]), np.diff(oriented_mags))) # make sure non-cummulative between bins

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


def scale_and_invert(image, maxValue = 255):
	"""
	scales and inverts a [0,255] (uint8) image type to be within [0,1] with same number of bins
	"""
	return(1/maxValue * cv2.bitwise_not(image))


##  ----------------------------Below this line are outdated functions---------------------------------------------------------------------------------------------------


def centroid(cnt):
	"""
	Gives the position of the center of mass of a given grayscale image based on first-order moments

	Parameters:
	------------
	cnt: list
		list of x-y coordinate pairs defining the contour of an object
	Returns:
	------------
	centroid: numpy array
		x,y coordinate of centroid in image
	"""
	M = cv2.moments(cnt)
	centroid = (int(round(M['m10']/M['m00'])), int(round(M['m01']/M['m00'])))

	return(centroid)





def get_Orientation(image, moments, draw=True):
	"""
	Gives the orientation angle and axe lengthes of a given object relative to a typical horizontal axis by 2nd-order image moments

	Parameters:
	------------
	image: numpy array 
		two-dimensional image array to be drawn on
	moments: dictionary 
		dictionary of moments for a given grayscale image 
	draw: boolean (default: True)
		determines if the principal will be drawn on image

	Returns:
	------------
	theta: float
		orientation angle of image object relative to typical horizontal axis
	axes: tuple (MA, ma)
		maximum(MA) and minimum(ma) principal axe lengths 
	"""


	M = moments
	c = centroid(image, M, draw)

	# pseudo-normalized central moments
	mu11_p = M['mu11']/M['m00']
	mu20_p = M['mu20']/M['m00']
	mu02_p = M['mu02']/M['m00']

	# principal axe lengths
	alpha = (mu20_p + mu02_p)/2
	beta = np.sqrt((4*(mu11_p**2) + (mu20_p-mu02_p)**2)/2)
	lamdas = (alpha+beta, alpha-beta)
	MA = np.max(np.abs(lamdas))
	ma = np.min(np.abs(lamdas))

	# orientation angle
	theta = .5 * np.arctan(2 * mu11_p / (mu20_p - mu02_p + np.finfo(float).eps))

	# Draw Orinetation Lines 
	theta = 1 + np.floor(theta * 180/np.pi)
	axes = (MA, ma)
	if draw is True:
		## find max-axis point
		max_line_x = c[0] + 14 * np.cos(theta*np.pi/180)
		max_line_y = c[1] + 14 * np.sin(theta*np.pi/180)
		max_pt = (int(max_line_x), int(max_line_y))

		## find min-axis point
		min_line_x = c[0] + 3 * np.cos(theta*np.pi/180-(np.pi/2))
		min_line_y = c[1] + 3 * np.sin(theta*np.pi/180-(np.pi/2))
		min_pt = (int(min_line_x), int(min_line_y))

		cv2.line(image, c, max_pt, (0), 1)
		cv2.line(image, c, min_pt, (0), 1)

	return(theta, axes)



def boundingRectangle(epsilon, cnt=None, x=None, y=None, MA=None, ma=None, angle=None):
	"""
	Creates the dimensions of a bounding box based off of the dimensions of the principal axes and orientation
	
	TODO: raise error if angle not in radians

	Parameters:
	----------
	cnt: list
		list of x-y coordinates that describe a contour of the object. Use this or specify the following
	x: int 
		x-coordinate of center point
	y: int
		y-coordinate of center point
	MA: int
		maximum fitted ellipse axis length
	ma: int
		minimum fitted ellipse axis length
	angle: float
		orientation angle with respect to typical horizontal axis (radians)
	epsilon: float
		noise value to extend bounding box by small amount

	Returns:
	----------
	x: int 
		x-coordinate of center point
	y: int
		y-coordinate of center point
	base: int
		length of base of bounding box
	height: int
		length of height of bounding box
	br_pt: tuple
		x-y coordinate of top-left corner of bounding box
	"""

	## fit ellipse
	if cnt is not None:
		(x,y), (ma,MA), angle = cv2.fitEllipse(cnt)
		angle = angle*np.pi/180

	## find base, height, top-left corner of bounding box relative to center point.
	base = np.abs(ma * np.cos(angle)) + np.abs(MA * np.cos(angle + np.pi/2)) + epsilon
	base = int(round(base))
	height = np.abs(ma * np.sin(angle)) + np.abs(MA * np.sin(angle + np.pi/2)) + epsilon
	height = int(round(height))
	br_pt = (int(round(x - base/2)), int(round(y - height/2)))
	angle = angle*180/np.pi

	return(x, y, MA, ma, angle, base, height, br_pt)
	




def scaleContour(cnt, x, y, cnt_scale):
	"""
	Linearly scales a contour by given scale factor

	Parameters:
	-------------
	cnt: list
		list of x-y coordinates that describe a contour of the object 
	x: int
		x-coordinate of center point 
	y: int
		y-coordinate of center point
	cnt_scale: float
		linear scale factor

	Returns:
	-------------
	cnt_scaled: list
		list of x-y coordinates that describe a scaled contour of the object 
	"""

	## Define scale and translation matix
	scale_matrix = np.array([[cnt_scale, 0],[0, cnt_scale]])
	trans_matrix = np.array([x, y]) 

	## Linearly scale contour
	cnt_scaled = np.array([trans_matrix + (pnt - trans_matrix) @ scale_matrix for pnt in cnt]).astype(int)

	return(cnt_scaled)



def ellipseDiameter(theta, MA, ma):
	"""
	find diameter of an ellipse given a angle theta away from its major axis
	"""
	d = (MA*ma) / np.sqrt((ma*np.cos(theta))**2 + (MA*np.sin(theta))**2)

	return(d)





