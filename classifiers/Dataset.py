import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os 


class Dataloader:
	def __init__(self, path_to_images, break_percent, resize_shape, network_type='CNN', one_hot=True):
		self.wb_images = np.load(path_to_images).item()
		self.resize_shape = resize_shape
		self.ONE_HOT = one_hot 
		self.NETWORK_TPE = network_type

		num_images = len(self.wb_images['label'])
		num_train = int(break_percent * num_images)

		indices = np.arange(0, num_images)
		np.random.shuffle(indices)

		for key in self.wb_images.keys():
			if key == "full" or key == "label":
				self.wb_images[key] = np.array(self.wb_images[key])[indices]

		self.train = {"images":[], "labels":[]}
		self.val = {"images":[], "labels":[]}

		print("Train:")
		self.processes_images(self.train, 0, num_train)

		print("Validation:")
		self.processes_images(self.val, num_train, num_images+1)


	def processes_images(self, data_type, start, end, inv_lim = 60, interpolation_method=cv2.INTER_LINEAR):
		for i, image in enumerate(self.wb_images["full"][start:end]):
			# if np.mean(image) <= inv_lim:
			# 	image = cv2.bitwise_not(image)

			if self.NETWORK_TPE == 'CNN':
				if np.any(np.array(image.shape) == 0):
					continue
				resize = cv2.resize(image, self.resize_shape, interpolation_method)
				normalize = resize / 255

			elif self.NETWORK_TPE == 'MLP':
				image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				resize = cv2.resize(image_gray, self.resize_shape, interpolation_method)
				reshape = resize.reshape(resize.size)
				normalize = reshape/255
			else:
				raise NameError(NETWORK_TYPE + " is not a recognized NETWORK_TYPE. Available options: 'MLP', CNN'.")

			data_type["images"].append(normalize)

			if self.ONE_HOT:
				if self.wb_images["label"][start+i] == 0:
					data_type["labels"].append(np.array([1,0]))
				else:
					data_type["labels"].append(np.array([0,1]))
			else:
				data_type["labels"].append(self.wb_images["label"][start+i])

		for key in data_type.keys():
			data_type[key] = np.array(data_type[key])

			print('	Object {}'.format(key))
			print('		Shape {}'.format(data_type[key].shape))



class Dataset:
	def __init__(self, data):
		self.data = data
		self.index_in_epoch = 0
		self.epochs_completed = 0
		self.num_examples = data["images"].shape[0]
		
	def next_batch(self, batch_size):
		start = self.index_in_epoch
		# go to the next batch
		if start + batch_size > self.num_examples:
			self.epochs_completed += 1
			rest_num_examples = self.num_examples - start
			rest_images = self.data["images"][start:self.num_examples]
			rest_labels = self.data["labels"][start:self.num_examples]

			start = 0
			self.index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
			end =  self.index_in_epoch  
			new_part_images = self.data["images"][start:end]
			new_part_labels = self.data["labels"][start:end]

			return np.concatenate((rest_images, new_part_images), axis=0),  np.concatenate((rest_labels, new_part_labels), axis=0)
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			return self.data["images"][start:end], self.data["labels"][start:end]

