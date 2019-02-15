import tensorflow as tf 
import numpy as np 

class WB_Classifier(object):

	def __init__(self, input_dim, output_dim, reshape_size=13*25*64, hidden_dim=1000, NETWORK_TYPE='CNN', name="image", TRAIN=False, data_dict=None):
		self.x_dim = input_dim
		self.y_dim = output_dim
		self.h_dim = hidden_dim
		self.TRAIN = TRAIN
		self.data_dict = data_dict
		self.reshape_size = reshape_size        
		
		self.keep_prob = tf.placeholder(tf.float32)
		self.y_ = tf.placeholder(tf.float32, [None, self.y_dim])

		if NETWORK_TYPE == 'CNN':
			self.x = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], 3])
			with tf.variable_scope(name):
				self.CNN()
		elif NETWORK_TYPE == 'MLP':
			self.x = tf.placeholder(tf.float32, [None, self.x_dim])
			with tf.variable_scope(name):
				self.MLP()
		else:
			raise NameError(NETWORK_TYPE + " is not a recognized NETWORK_TYPE. Available options: 'MLP', CNN'.")

		self.loss = self.loss_function()
		self.accuracy = self.accuracy_function()

	def CNN(self):
		# First conv layer
		conv1 = self.conv_layer(self.x, 5, 5, 32, 1, 1, padding="SAME", name='conv1')
		mp1 = max_pool(conv1, 2, 2, 2, 2, padding="SAME", name='mp1')

		# Second conv layer
		conv2 = self.conv_layer(mp1, 5, 5, 64, 1, 1, padding="SAME", name="conv2")
		mp2 = max_pool(conv2, 2, 2, 2, 2, padding="SAME", name='mp2')
		mp2_reshaped = tf.reshape(mp2, [-1, self.reshape_size])

		# First densely connected layer
		fc3 = self.fc_layer(mp2_reshaped, self.reshape_size, 1024, name='fc3')
		dropout3 = dropout(fc3, self.keep_prob)

		# read-out layer
		self.y = self.fc_layer(dropout3, 1024, self.y_dim, name='read-out', RELU=False)


	def MLP(self):
		h = fc_layer(self.x, self.x_dim, self.h_dim, name='hidden')
		self.y = fc_layer(h, self.h_dim, self.y_dim, name='out', RELU=False)

	def loss_function(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

	def accuracy_function(self):
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def conv_layer(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding, RELU=True):
		channels = int(x.get_shape()[-1])

		with tf.variable_scope(name) as scope:
			if self.TRAIN:
				weights = tf.get_variable('weights', shape=[filter_height, filter_width, channels, num_filters], initializer=tf.truncated_normal_initializer(stddev=0.01))
				biases = tf.get_variable('biases', initializer = tf.constant(0.1, shape=[num_filters]))#shape=[num_filters], initializer=tf.truncated_normal_initializer(stddev=0.01))
			else:
				weights = self.get_weights(name)
				biases = self.get_biases(name)

		conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding=padding) + biases

		if RELU:
			return tf.nn.relu(conv)
		else:
			return conv

	def fc_layer(self, x, input_dim, output_dim, name, RELU=True):
		with tf.variable_scope(name) as scope:
			if self.TRAIN:
				weights = tf.get_variable('weights', shape=[input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
				biases = biases = tf.get_variable('biases', initializer = tf.constant(0.1, shape=[output_dim]))#tf.get_variable('biases', shape = [output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
			else:
				weights = self.get_weights(name)
				biases = self.get_biases(name)

		out = tf.matmul(x,weights) + biases

		if RELU:
			return tf.nn.relu(out)
		else:
			return out 

	def get_weights(self, name):
		return tf.constant(self.data_dict[name][0])
		
	def get_biases(self, name):
		return tf.constant(self.data_dict[name][1])


def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding, name):
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides =[1, stride_y, stride_x, 1], padding=padding, name=name)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)
