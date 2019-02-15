import tensorflow as tf
import numpy as np


class AlexNet(object):
    """
    Class encapsulating an implementation of AlexNet in tensorflow
    """

    def __init__(self, input_dim, output_dim, name, path_to_ImageNet_weights=None, discarded_layers = [], data_dict=None, TRAIN=True):
        """
        Inputs:
        ------------
        input_dim: tuple 
            input dimension to the network
        output_dim: int
            output dimension of the classification
        name: str
            the name of the classifier
        path_to_weight: str
            path to the pre-trarianed weights filr

        """
        self.training_name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.TRAIN = TRAIN
        self.PATH_TO_IMAGENET_WEIGHTS = path_to_ImageNet_weights
        self.data_dict = data_dict
        self.discarded_layers = discarded_layers

        self.x = tf.placeholder(tf.float32, [None, input_dim[0], input_dim[1], 3])
        self.y_ = tf.placeholder(tf.float32, [None, output_dim])
        self.keep_prob = tf.placeholder(tf.float32)

        # build the computational graph of AlexNet
        with tf.variable_scope(name):
            self.build_graph()

        self.loss = self.loss_function()
        self.accuracy = self.accuracy_function()

    def build_graph(self):
        # 1st Layer (conv1)
        conv1 = self.conv_layer(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        lrn1 = lrn(conv1, 2, 1, 1e-05, 0.75, name='lrn1')
        pool1 = max_pool(lrn1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: (conv2)
        conv2 = self.conv_layer(pool1, 5, 5, 256, 1, 1, padding='SAME', splits=2, name='conv2')
        lrn2 = lrn(conv2, 2, 1, 1e-05, 0.75, name='lrn2')
        pool2 = max_pool(lrn2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: (conv3)
        conv3 = self.conv_layer(pool2, 3, 3, 384, 1, 1, padding='SAME', name='conv3')

        # 4th Layer: (conv4)
        conv4 = self.conv_layer(conv3, 3, 3, 384, 1, 1, padding='SAME', splits=2, name='conv4')

        # 5th Layer: (conv5)
        conv5 = self.conv_layer(conv4, 3, 3, 256, 1, 1, padding='SAME', splits=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: (fc6)
        pool5_reshaped= tf.reshape(pool5, [-1, 2*5*256])
        fc6 = self.fc_layer(pool5_reshaped, 2*5*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: (fc7)
        fc7 = self.fc_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        # Readout layer (fc8)
        self.y = self.fc_layer(dropout7, 4096, self.output_dim, RELU=False, name='fc8')

    def loss_function(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

    def accuracy_function(self):
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_ImageNet_weights(self, session):
        """
        INPUTS:
        -----------
        session: tensorflow session object
            the tf session initialized for training
        """
        # Note: The pre-trained weights for this AlexNet implementation are required to be split for conv2, conv4, & conv5 layers. 
        pre_trained_weights = np.load(self.PATH_TO_IMAGENET_WEIGHTS, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        with tf.variable_scope(self.training_name):
            for name in pre_trained_weights:
                if name not in self.discarded_layers:
                    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                        for data in pre_trained_weights[name]:
                            # weights
                            if len(data.shape) != 1:
                                var = tf.get_variable('weights', trainable=True)
                                session.run(var.assign(data))

                            # biases
                            else:
                                var = tf.get_variable('biases', trainable=True)
                                session.run(var.assign(data))

    def get_weights(self, name):
        return tf.constant(self.data_dict[name][0])

    def get_biases(self, name):
        return tf.constant(self.data_dict[name][1])


    def conv_layer(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding, splits=1, RELU=True):
        channels = int(x.get_shape()[-1])

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            if self.TRAIN:
                if name in self.discarded_layers:
                    weights = tf.get_variable('weights', shape=[filter_height, filter_width, channels/splits, num_filters], initializer=tf.truncated_normal_initializer(stddev=0.01))
                    biases = tf.get_variable('biases', initializer = tf.constant(0.1, shape=[num_filters]))

                else:
                    weights = tf.get_variable('weights', shape=[filter_height, filter_width, channels/splits, num_filters])
                    biases = tf.get_variable('biases', shape=[num_filters])
            else:
                weights = self.get_weights(name)
                biases = self.get_biases(name)

        if splits == 1:
            conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding=padding) + biases

        else:
            # Note: The pre-trained weights for this AlexNet implementation are required to be shared and split for conv2, conv4, & conv5 layers
            input_groups = tf.split(axis=3, num_or_size_splits=splits, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=splits, value=weights)
            output_groups = [tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups) + biases

        if RELU:
            return tf.nn.relu(conv)
        else:
            return conv

    def fc_layer(self, x, input_dim, output_dim, name, RELU=True):
        with tf.variable_scope(name) as scope:
            if self.TRAIN:
                if name in self.discarded_layers:
                    weights = tf.get_variable('weights', shape=[input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
                    biases = tf.get_variable('biases', initializer = tf.constant(0.1, shape=[output_dim]))
                else:
                    weights = tf.get_variable('weights', shape=[input_dim, output_dim])
                    biases = tf.get_variable('biases', shape=[output_dim])
            else:
                weights = self.get_weights(name)
                biases = self.get_biases(name)

            out = tf.matmul(x,weights) + biases

        if RELU:
            return tf.nn.relu(out)
        else:
            return out 

def lrn(x, depth_radius, bias, alpha, beta, name):
    return tf.nn.local_response_normalization(x, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding, name):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides =[1, stride_y, stride_x, 1], padding=padding, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
