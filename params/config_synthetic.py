import cv2 
import params.config as config
#from generate_synthetic_dataset import scale

#gathering dataset
path_to_images = config.dataset_dir + '/bounding_box_labeled_images.npy' 
x_lim = (10, 1000)
y_lim = (5, 500)
padding_size = 20


# generated_copies
pre_load = True
copy_chance = .15

# Scale
def get_args(name):
	if name == "scale":
		transformation_name = "Linear Interpolation" 
		transformation_args = [(1.25,1.25,cv2.INTER_LINEAR), 
							   (1.5,1.5,cv2.INTER_LINEAR),
							   (1.75,1.75,cv2.INTER_LINEAR),
							   (2.0,2.0,cv2.INTER_LINEAR)]
		labels = ["1.25", "1.5", "1.75", "2.0"]

	elif name == "rotate":
		transformation_name = "Rotation" 
		transformation_args = [3, 6, 9, 12]
		labels = ["3", "6", "9", "12"]

	elif name == "contrast":
		transformation_name = "Linear Contrast" 
		transformation_args = [.6, .8, 1.2, 1.4]
		labels = [".6", ".8", "1.2", "1.4"]

	elif name == "noise":
		transformation_name = "White Noise" 
		transformation_args = [3, 9, 21]
		labels = ["3", "9", "21"]

	elif name == "blur":
		transformation_name = "Gaussian Blur" 
		transformation_args = [(3,3), (9,9), (21,21)]
		labels = ["3", "9", "21"]

	elif name == "reflect":
		transformation_name = "Reflection" 
		transformation_args = [0, 1, '180']
		labels = ["x-axis", "y-axis", "180"]

	return transformation_name, transformation_args, labels