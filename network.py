import numpy as np
import tensorflow as tf


def create_features_tensor(num_features,  graph):
    with graph.as_default():
        feature_vector = tf.placeholder(
            shape=[num_features, None], dtype=tf.float32, name='features')
       

        return feature_vector

def target_network_variables_placeholders(num_layers, layers_size, graph):
    """Returns tuples with the weights and biases"""

    with graph.as_default():
        assert(num_layers == len(layers_size)), "Num of hidden layers : {} is not the same as the number of defined layers sizes {}".format(
            num_layers, len(layers_size))

        weights = []
        biases = []

        for i in range(num_layers - 1):

            weights.append(tf.placeholder(
                name="w"+str(i), shape=[layers_size[i+1], layers_size[i]], dtype=tf.float32))
            biases.append(tf.get_variable(
                name="b"+str(i), shape=[layers_size[i+1], 1], dtype=tf.float32))
            
        weights = tuple(weights)
        biases = tuple (biases)

        return weights, biases


def create_variables(num_layers, layers_size, graph):
    """" Return:
                weights tensor
                bias    tensor"""

    with graph.as_default():
        assert(num_layers == len(layers_size)), "Num of hidden layers : {} is not the same as the number of defined layers sizes {}".format(
            num_layers, len(layers_size))

        weights = []
        biases = []

        for i in range(num_layers - 1):

            weights.append(tf.get_variable(
                name="w"+str(i), shape=[layers_size[i+1], layers_size[i]], dtype=None, initializer=tf.contrib.layers.xavier_initializer()))
            biases.append(tf.get_variable(
                name="b"+str(i), shape=[layers_size[i+1], 1], dtype=None, initializer=tf.zeros_initializer()))

        return weights, biases


def forward_propagation(features, weights, biases, activations, graph):
    """
    Returns tensor for output layer
    """
    with graph.as_default():

        num_activations = len(weights)
        assert(num_activations == len(activations)), "You need {} function activations and you provided {}".format(
            num_activations, len(activations))

        value = features

        for w, b, activation in zip(weights, biases, activations):
            value = tf.matmul(w, value) + b

            if activation is not None:
                value = activation(value)

        output_pred = value

        return output_pred
