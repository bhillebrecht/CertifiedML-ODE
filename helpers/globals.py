###################################################################################################
# Copyright (c) 2022 Birgit Hillebrecht
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
###################################################################################################

import logging
import sys

import tensorflow as tf

#########################################################################################################
## This files keeps track and handles global variables 
#########################################################################################################

PREFIX = 'tanh_'
ACTIVATION_FUNCTION = tf.nn.tanh
OPTIMIZER = 'adam'

def get_activation_function():
    """
    Returns tf.nn activation function
    """
    return ACTIVATION_FUNCTION

def get_prefix() -> str:
    return PREFIX

def set_activation_function(af) -> None:
    """
    Sets activation function for next NN.

    :param string af: activation function by string, supported are tanh, silu, gelu and softmax
    """
    global ACTIVATION_FUNCTION
    global PREFIX
    PREFIX = af + '_'

    if af == 'tanh':
        ACTIVATION_FUNCTION = tf.nn.tanh
    elif af == 'silu':
        ACTIVATION_FUNCTION = tf.nn.silu
    elif af == 'gelu':
        ACTIVATION_FUNCTION = tf.nn.gelu
    elif af == 'softmax':
        ACTIVATION_FUNCTION = tf.nn.softmax
    else:
        logging.error("The activation function given \'" + af + "\' is invalid. Value must be either tanh, silu, gelu or softmax")
        sys.exit()

def set_optimizer(o) -> None:
    """
    Sets optimizer for next NN.

    :param string o: optimizer by string, supported are lbfgs and adam
    """
    global OPTIMIZER
    if not (o == 'lbfgs') and not (o =='adam'):
        logging.error("The optimizer chosen is not supported: \'"+o+"\'. Valid values are lbfgs and adam.")
        sys.exit()

    OPTIMIZER = o

def get_optimizer() -> str:
    global OPTIMIZER
    return OPTIMIZER