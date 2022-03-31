###################################################################################################
# Copyright (c) 2021 Jonas Nicodemus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################
#
# This file incorporates work and modifications to the originally published code
# according to the previous license by the following contributors under the following licenses
#
#   Copyright (c) 2022 Birgit Hillebrecht
#
#   This code has been developed as part of [TBD insert link to pub]
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
###################################################################################################

import abc

import tensorflow as tf
from keras.backend import get_value

from base.nn import NN


class PINN(NN, metaclass=abc.ABCMeta):
    """
    Class used to represent a Physics informed Neural Network, children of NN.
    """

    def __init__(self, layers, lb, ub):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        super().__init__(layers, lb, ub)

        self.loss_object = self.loss

    def loss(self, y, y_pred):
        """
        Customized loss object to represent the composed mean squared error
        for Physics informed Neural Networks.
        Consists of the mean squared error 
            between the predictions from the DNN (model) and reference values of the solution from the differential equation
            between and the mean squared error of the predictions of the PINN (f_model) with zero.

        :param tf.tensor y: reference values of the solution of the differential equation
        :param tf.tensor y_pred: predictions of the solution of the differential equation
        :return: tf.tensor: composed mean squared error value
        """

        w_data = 1
        w_phys = 1

        f_pred = self.f_model()
        L_data = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), axis=1))
        L_phys = tf.reduce_mean(tf.reduce_sum(tf.square(f_pred), axis=1))
        
        L = w_data * L_data + w_phys * L_phys

        return L

    def f_model(self, x):
        """
        Declaration of the function for the implementation of the f_model for a specific differential equation.
        """
        pass

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        if x.shape[1] == self.input_dim :
            return self.model.predict(x), self.f_model(x)
        else :
            return self.model.predict(x[:,0:self.input_dim]), self.f_model(x[:,0:self.input_dim])
