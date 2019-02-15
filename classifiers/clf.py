import numpy as np 
import tensorflow as tf

from classifiers.classifier import WB_Classifier
from classifiers.alexnet_binary import AlexNet

class wb_clf(object):
	def __init__(self, clf_filename, input_dim, reshape_size, name, ALEXNET=False):
		data_dict = np.load(clf_filename).item()
		if ALEXNET:
			self.model = AlexNet(input_dim, 2, name=name, TRAIN=False, data_dict=data_dict)
		else:
			self.model = WB_Classifier(input_dim, 2, reshape_size=reshape_size, NETWORK_TYPE='CNN', name=name, TRAIN=False, data_dict=data_dict)

	def eval(self, images):
		output = self.model.y.eval(feed_dict={self.model.x:images, self.model.keep_prob:1})
		classification = np.argmax(output, 1)
		return(classification[0])
     


