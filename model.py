from keras.models import load_model
from keras.layers import Concatenate, Softmax, Reshape, Multiply, Layer
from keras.saving import register_keras_serializable
import tensorflow as tf


@register_keras_serializable(package="ensemble")
class StackBranches(Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)
    def compute_output_shape(self, input_shape):
        n = len(input_shape)          # number of branches
        return (input_shape[0][0], n, input_shape[0][1])

@register_keras_serializable(package="ensemble")
class WeightedSum(Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def load_classifier(path):
    return load_model(path, safe_mode=False, compile=False)

def predict(model, X):
    return model.predict(X, verbose=0)
