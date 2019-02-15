import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import params.config as config
import params.config_synthetic as config_synthetic

def scale(image, arguments):
	fx, fy, interpolation = arguments
	scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)
	return scaled_image

def rotate(image, arguments):
	theta = arguments
	# image = cv2.copyMakeBorder(image,4,4,4,4, cv2.BORDER_REPLICATE)
	row, col = image.shape
	rotation_matrix = cv2.getRotationMatrix2D((col/2, row/2), theta, 1) # find rotation matrix
	rotated_image = cv2.warpAffine(image, rotation_matrix, (col, row), borderMode=cv2.BORDER_REPLICATE) # rotate image by trasforming with rotation matrix
	return rotated_image

def contrast_adjust(image, arguments):
	alpha = arguments
	#brightness_array  = np.uint8(beta * np.ones(image.shape))
	adjContrast_image = cv2.scaleAdd(image, alpha, np.zeros_like(image, dtype=np.uint8))
	return adjContrast_image

def blur(image, arguments):
	kernel = arguments
	return cv2.GaussianBlur(image, kernel, 0)

def add_noise(image, arguments):
	std = arguments
	noise = np.random.normal(0, std, image.shape)
	noisy_iamge = np.maximum(np.minimum(image + noise, 255), 0) # cv2.scaleAdd(image, 1, np.uint8(noise))
	return np.uint8(noisy_iamge)

def reflect(image, arguments):
	axis = arguments
	if axis == '180':
		return np.uint8(np.rot90(image,2))
	else:
		return np.uint8(np.flip(image, axis=axis))


class GenerateSyntheticData(object):

	def __init__(self):
		self.gathered = {"image":[], "mark":[]} 

	def gather(self, path_to_images, x_lim, y_lim, padding_size):
		"""
		Gathers blots from ground truth labeled images

		Parameters:
		-------------
		path_to_images: str
			file name containing the images
		x_lim: tuple of size 2
			contrains the (min, max) x-dimensional limits for gathering segmented blots from images
		y_lim: tuple of size 2
			contrains the (min, max) y-dimensional limits for gathering segmented blots from images
		padding_size: int
			size to pad the blot image
		
		"""
		figure_data = np.load(path_to_images).item() # load figure data pointers

		for image, rects in zip(figure_data["image"], figure_data["rects"]):
			for r in rects:
				gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
				img_section = gray[r[1]:r[3], r[0]:r[2]]
				mark = np.ones_like(img_section, dtype=np.uint8)
				y, x = img_section.shape
				if x < x_lim[0] or x > x_lim[1] or y < y_lim[0] or y > y_lim[1]:
					continue

				# apply padding and append the data
				img_section = cv2.copyMakeBorder(img_section, padding_size, padding_size, padding_size, padding_size,
												 cv2.BORDER_REPLICATE)
				mark = cv2.copyMakeBorder(mark, padding_size, padding_size, padding_size, padding_size,
										  cv2.BORDER_CONSTANT, value=(0))

				self.gathered["image"].append(img_section)
				self.gathered["mark"].append(mark)



		print("%s images were gathered"% (len(self.gathered["image"])))

	def generate_copies(self, copy_chance, transformation_name=None, transformation_args=None):
		"""
		generates synthetic dataset of blot images with a specified chance of duplication.

		Parameters:
		----------------
		copy_chance: float 
			chance each gathered blot will be duplicated in the dataset
		transformation_name: str
			function used to perform a pixel-wise transformation on the image
		tranformation_args: func args
			transformtation function arguments
		"""

		## initialize the synthetic data dictionary 
		synthetic_data = {"image":[], "mark":[], "label":[]}

		## draw from a binomial distrubution to determine if a given image should be copied
		copy_decision = np.random.binomial(1, copy_chance, len(self.gathered["image"]))

		# extract the images that were determined to be copied
		copies_img = [self.gathered["image"][j] for j,i in enumerate(copy_decision) if i==1]
		copies_mark = [self.gathered["mark"][j] for j,i in enumerate(copy_decision) if i==1]
		copy_indices = [j for j,i in enumerate(copy_decision) if i==1]

		assert np.all((copies_img[0] - self.gathered["image"][copy_indices[0]]) == 0)
		
		# transform the images if transformation function is given	
		transformation = self.get_transformation(transformation_name)	
		if transformation is not None:
			for c_idx, copy in enumerate(copies_img):
				copies_img[c_idx] = transformation(copy, transformation_args)
				if transformation_name == "scale" or transformation_name == "rotate" or transformation_name == "reflect": 
					copies_mark[c_idx] = transformation(copies_mark[c_idx], transformation_args)

		## compile all the sythetic data
		synthetic_data["image"] += self.gathered["image"] + copies_img
		synthetic_data["mark"] += self.gathered["mark"] + copies_mark
		synthetic_data["label"] += list(range(len(self.gathered["image"]))) + copy_indices

		print("Number of original images: %s; Number of copied images: %s"%(len(self.gathered["image"]), 
			len(synthetic_data["image"]) - len(self.gathered["image"])))

		return(synthetic_data)

	def get_transformation(self, name):
		if name == "scale":
			return scale
		elif name == "rotate":
			return rotate
		elif name == "contrast":
			return contrast_adjust
		elif name == "noise":
			return add_noise
		elif name == "blur":
			return blur
		elif name == "reflect":
			return reflect
		else:
			return None

