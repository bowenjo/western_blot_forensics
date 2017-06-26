import cv2
import os
import shutil
import numpy as np 
import wbFingerprintUtils as utils 
from sklearn.isotonic import IsotonicRegression


def rotateBlots(filename, maxRotationAngle, spacing):
	"""
	Creates a dictionary of images for all orientations defined by maxRotation and spacing.

	Parameters:
	-----------
	filename: str
		filename housing the image to be rotated
	maxRotationAngle: float (0-360 in degrees)
		maximum orientation angle to rotated the image
	spacing: int
		number of equal spaces between -maxRoatationAngle and maxRotationAngle
	interpolation: cv2 interpolation method
		method to interpolate pixels with rotation

	Returns:
	-----------
	wb_rotatedDatabase: dictionary

	"""

	## Load in the image
	image = cv2.imread(filename)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	row, col = np.shape(image_gray)

	## initialize rotation dictionary
	wb_rotatedDatabase = {"theta": [],
						   "image": [],
						   "original": image
						   }

	## perform rotation on each image defined by maxRotationAngle
	for theta in np.linspace(-maxRotationAngle, maxRotationAngle, spacing):
		rotation_matrix = cv2.getRotationMatrix2D((col/2, row/2), theta, 1) # find rotation matrix
		rotated_image = cv2.warpAffine(image, rotation_matrix, (col, row), borderValue = (255,255,255)) # rotate image by trasformin with rotation matrix

		wb_rotatedDatabase["theta"].append(theta)
		wb_rotatedDatabase["image"].append(rotated_image)

	return(wb_rotatedDatabase)




def scaleBlots(filename, maxScale, minScale, spacing, interpolation):
	"""
	Creates a dictionary of images for all scales defined by maxScale and spacing.

	Parameters:
	-----------
	filename: str
		filename housing the image to be scaled
	maxScale: float 
		maximum scale factor to scale the image
	spacing: int
		number of equal spaces between -maxScale and maxScale
	interpolation: cv2 interpolation method
		method to interpolate pixels with scaling

	Returns:
	-----------
	wb_scaledDatabase: dictionary

	"""

	## Load in the image
	image = cv2.imread(filename)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	row, col = np.shape(image_gray)

	## initialize scalining dictionary
	wb_scaledDatabase = {"scaleFactor" : [],
						 "image": [],
						 "original": image
						 }

	for sf in np.linspace(minScale, maxScale, spacing):
		scaled_row = int(round(sf * row)) # scale image row
		scaled_col = int(round(sf * col)) # scale image column

		resized_image = cv2.resize(image, (scaled_col, scaled_row), interpolation=interpolation) # resize image to appropriate scale

		wb_scaledDatabase["scaleFactor"].append(sf)
		wb_scaledDatabase["image"].append(resized_image)

	return(wb_scaledDatabase)



def adjustBlotContrastLinear(filename, maxContrast, minContrast, spacing, brightness):
	"""
	Creates a dictionary of images for spanning contrasts defined by maxContrast, minContast and spacing at constant brightness.

	Parameters:
	-----------
	filename: str
		filename housing the image to be adjusted
	maxContrast: float 
		maximum contrast scaler
	minContrast: float
		minumum contrast scaler
	spacing: int
		number of equal spaces between -minContrast and maxContrast
	brightness: float
		brightness factor (constant)

	Returns:
	-----------
	wb_adjContrastDatabase: dictionary

	"""

	## Load in the image
	image = cv2.imread(filename)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	## intitialize 

	wb_adjContrastDatabase = {"contrastFactor": [],
							  "image": [],
							  "original": image,
							  "brightness": brightness
							  }

	for cf in np.linspace(minContrast, maxContrast, spacing):
		brightness_array  = np.uint8(brightness * np.ones(image.shape))
		adjContrast_image = cv2.scaleAdd(image, cf, brightness_array)

		wb_adjContrastDatabase["contrastFactor"].append(cf)
		wb_adjContrastDatabase["image"].append(adjContrast_image)

	return(wb_adjContrastDatabase)



class AdjustBlotContrastNonLinear(object):

	def __init__(self, filename, num_curves, num_cntpoints, CUSTOM="random"):
		"""
		Creates a ditionary of images for spanning nonlinear contrasts as defined by contrast "curves"

		Parameters:
		------------
		filename: str
			filename housing the image to be adjusted
		num_curves: int
			gives the number of randomly generated contrast "curves"
		num_cntpoints: int
			gives the number of user control points on contrast "curves"

		Returns:
		-----------
		wb_adjContrastDatabase: dictionary
		
		"""
		if CUSTOM == "S2S_INV":
			ydata = self.generate_S_to_Sinv(25)
			num_cntpoints = ydata.shape[1] 
			num_curves = ydata.shape[0]
			xdata = np.linspace(0, 255, num_cntpoints)

		elif CUSTOM == "LOW2HIGH":
			ydata, xdata = self.generate_LOW_to_HIGH(25)
			num_curves = ydata.shape[0]

		elif CUSTOM == 'ML2EX':
			ydata, xdata = self.generate_ML_to_EX(25)
			num_curves = ydata.shape[0]

		elif CUSTOM == "random":
			xdata = np.linspace(0, 255, num_cntpoints)
			ydata = np.sort(np.random.rand(num_curves, int(len(xdata))) * 255, axis=1)

		## Load in image
		image = cv2.imread(filename)
		shape = image.shape
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		## intialize

		self.wb_adjContrastDatabase = {"distance": [],
								       "image": [],
								  	   "original": image,
								  	   "ydata": ydata,
								  	   "xdata": xdata}

		for i in range(num_curves):
			y = ydata[i]
			if CUSTOM == "LOW2HIGH" or CUSTOM == "ML2EX":
				x = xdata[i]
			else:
				x = xdata
			ir = IsotonicRegression()
			ir.fit(x, y)
			y_ = ir.transform(image.flatten()).reshape(shape)
			self.wb_adjContrastDatabase["image"].append(np.uint8(y_))
			self.wb_adjContrastDatabase["distance"].append(np.linalg.norm(y-x))


	def generate_S_to_Sinv(self, num_datapoints):
		col1 = np.zeros((num_datapoints, 1)) * 255
		col2 = np.linspace(0, .5, num_datapoints).reshape(num_datapoints,1) * 255
		col3 = .5 * np.ones((num_datapoints, 1)) * 255
		col4 = np.linspace(.5, 1, num_datapoints)[::-1].reshape(num_datapoints,1) * 255
		col5 = np.ones((num_datapoints,1)) * 255
		return(np.hstack((col1, col2, col3, col4, col5)))

	def generate_LOW_to_HIGH(self, num_datapoints):
		col1 = np.zeros((num_datapoints,1)) * 255
		col2 = np.append(np.append(np.zeros(int(num_datapoints/2)), np.array([.5])), np.ones(int(num_datapoints/2))).reshape(num_datapoints,1) * 255
		col3 = np.ones((num_datapoints,1)) * 255

		col1x = np.zeros((num_datapoints,1)) * 255
		col2x = np.append(np.append(np.linspace(0,.6,int(num_datapoints/2)+1)[::-1][:int(num_datapoints/2)], np.array([.5])), np.linspace(.4,1,int(num_datapoints/2)+1)[::-1][1:]).reshape(num_datapoints,1) * 255
		col3x = np.ones((num_datapoints,1)) * 255
		return(np.hstack((col1, col2, col3)), np.hstack((col1x, col2x, col3x)))

	def generate_ML_to_EX(self, num_datapoints):
		col1 = np.zeros((num_datapoints,1)) * 255
		col2 = np.append(np.ones(int(num_datapoints/2)+1)*.5, np.zeros(int(num_datapoints/2))).reshape(num_datapoints,1) * 255
		col3 = np.append(np.ones(int(num_datapoints/2)+1)*.5, np.ones(int(num_datapoints/2))).reshape(num_datapoints,1) * 255
		col4 = np.ones((num_datapoints,1)) * 255
		
		col1x = np.zeros((num_datapoints,1)) * 255
		col2x = np.append(np.linspace(.25,.5, int(num_datapoints/2)+1), np.linspace(0,.49, int(num_datapoints/2))).reshape(num_datapoints,1) * 255
		col3x = np.append(np.linspace(.75, .5, int(num_datapoints/2)+1), np.linspace(1,.51, int(num_datapoints/2))).reshape(num_datapoints,1) * 255
		col4x = np.ones((num_datapoints,1)) * 255
		return(np.hstack((col1, col2, col3, col4)), np.hstack((col1x, col2x, col3x, col4x)))



## ----------- Execution Functions ------------------------------------------------------------------------------------

def create_directory(directory):
	if not os.path.exists(directory):
		os.mkdir(directory)
	else:
		shutil.rmtree(directory)
		os.mkdir(directory)


def main():

	wb_dir = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots"

	## label output directories
	wb_scale_output_dir = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_scale"
	wb_rot_output_dir = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_rotate"
	wb_adjConLinear_output_dir = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_adjConLinear"
	#wb_adjConNonLinear_output_dir = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_adjConNonLinear"

	dir_list = [wb_scale_output_dir, wb_rot_output_dir, wb_adjConLinear_output_dir] #, wb_adjConNonLinear_output_dir]

	# create output directories
	for directory in dir_list:
		create_directory(directory)


	## Save output to directories
	for subdirs, dirs, files in os.walk(wb_dir):
		for file in files: 
			path_to_file = os.path.join(wb_dir,file)

			np.save(wb_rot_output_dir + '/' + os.path.join(file) + "_rotated", rotateBlots(path_to_file, maxRotationAngle = 9, spacing = 11))
			np.save(wb_scale_output_dir + '/' + os.path.join(file) + "_scaled", scaleBlots(path_to_file, maxScale = 1.5, minScale=.5, spacing=11, interpolation=cv2.INTER_NEAREST))
			np.save(wb_adjConLinear_output_dir + '/' + os.path.join(file) + "_contrast_l", adjustBlotContrastLinear(path_to_file, maxContrast = 1.5, minContrast=.5, spacing=11, brightness=0))
			#np.save(wb_adjConNonLinear_output_dir + '/' + os.path.join(file) + "_contrast_nl", AdjustBlotContrastNonLinear(path_to_file, num_curves=9, num_cntpoints=5, CUSTOM="S2S_INV").wb_adjContrastDatabase)

def NL_Contrast_Builder():

	wb_dir = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots"

	# label output directories
	wb_acnl_rand = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_adjConNonLinear/acnl_rand"
	wb_acnl_S2S_INV = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_adjConNonLinear/acnl_S2S_INV"
	wb_acnl_LOW2HIGH = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_adjConNonLinear/acnl_LOW2HIGH"
	wb_acnl_ML2EX = "C:/Users/Joel/Documents/GitHub/NonRepositories/westernBlotForensics/wbFingerprints/HOG_testblots_transformed/HOG_testblots_adjConNonLinear/acnl_ML2EX" 

	dir_list = [wb_acnl_rand, wb_acnl_S2S_INV, wb_acnl_LOW2HIGH, wb_acnl_ML2EX]

	# create output directories
	for directory in dir_list:
		create_directory(directory)

	# sace output to directories
	for subdirs, dirs, files in os.walk(wb_dir):
		for file in files:
			path_to_file = os.path.join(wb_dir, file)

			np.save(wb_acnl_rand + '/' + os.path.join(file) + "_acnl_rand", 
				AdjustBlotContrastNonLinear(path_to_file, num_curves=100, num_cntpoints=5, CUSTOM="random").wb_adjContrastDatabase)
			np.save(wb_acnl_S2S_INV + '/' + os.path.join(file) + "_acnl_S2S_INV", 
				AdjustBlotContrastNonLinear(path_to_file, num_curves=None, num_cntpoints=None, CUSTOM="S2S_INV").wb_adjContrastDatabase)
			np.save(wb_acnl_LOW2HIGH + '/' + os.path.join(file) + "_acnl_LOW2HIGH", 
				AdjustBlotContrastNonLinear(path_to_file, num_curves=None, num_cntpoints=None, CUSTOM="LOW2HIGH").wb_adjContrastDatabase)
			np.save(wb_acnl_ML2EX + '/' + os.path.join(file) + "_acnl_ML2EX",
				AdjustBlotContrastNonLinear(path_to_file, num_curves=None, num_cntpoints=None, CUSTOM="ML2EX").wb_adjContrastDatabase)

if __name__ == '__main__':
	#main()
	NL_Contrast_Builder()










