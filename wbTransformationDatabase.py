import cv2
import os
import shutil
import numpy as np 
import wbFingerprintUtils as utils 


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



def adjustBlotContrast(filename, maxContrast, minContrast, spacing, brightness):
	"""
	Creates a dictionary of images for spanning contrasts defined by maxContrast, minContast and spacing at constant brightness.

	Parameters:
	-----------
	filename: str
		filename housing the image to be scaled
	maxContrast: float 
		maximum contrast scaler
	minContrast: float
		minumum contrast scaler
	spacing: int
		number of equal spaces between -maxScale and maxScale
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





def main():

	wb_dir = "C:/Users/Joel/Documents/GitHub/westernBlotForensics/wbFingerprints/HOG_testblots"

	## Create scale directory
	wb_scale_output_dir = "C:/Users/Joel/Documents/GitHub/westernBlotForensics/wbFingerprints/HOG_testblots_scale"
	if not os.path.exists(wb_scale_output_dir):
		os.mkdir(wb_scale_output_dir)
	else:
		shutil.rmtree(wb_scale_output_dir)
		os.mkdir(wb_scale_output_dir)
	
	## Create rotation directory
	wb_rot_output_dir = "C:/Users/Joel/Documents/GitHub/westernBlotForensics/wbFingerprints/HOG_testblots_rotate"
	if not os.path.exists(wb_rot_output_dir):
		os.mkdir(wb_rot_output_dir)
	else:
		shutil.rmtree(wb_rot_output_dir)
		os.mkdir(wb_rot_output_dir)

	## Create contrast-adjustment directory
	wb_adjCon_output_dir = "C:/Users/Joel/Documents/GitHub/westernBlotForensics/wbFingerprints/HOG_testblots_adjCon"
	if not os.path.exists(wb_adjCon_output_dir):
		os.mkdir(wb_adjCon_output_dir)
	else:
		shutil.rmtree(wb_adjCon_output_dir)
		os.mkdir(wb_adjCon_output_dir)

	## Save output to directories
	for subdirs, dirs, files in os.walk(wb_dir):
		for file in files: 
			path_to_file = os.path.join(wb_dir,file)

			np.save(wb_rot_output_dir + '/' + os.path.join(file) + "_rotated", rotateBlots(path_to_file, maxRotationAngle = 9, spacing = 11))
			np.save(wb_scale_output_dir + '/' + os.path.join(file) + "_scaled", scaleBlots(path_to_file, maxScale = 1.5, minScale=.5, spacing=11, interpolation=cv2.INTER_NEAREST))
			np.save(wb_adjCon_output_dir + '/' + os.path.join(file) + "_contrast", adjustBlotContrast(path_to_file, maxContrast = 1.5, minContrast=.5, spacing=11, brightness=0))


if __name__ == '__main__':
	main()










