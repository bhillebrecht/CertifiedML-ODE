###################################################################################################
# (c) 2022 Birgit Hillebrecht
# 
# This code has been developed as part of 
# 	  Certified machine learning: A posteriori error estimation for physics-informed neural networks
#     https://doi.org/10.48550/arXiv.2203.17055
# please kindly consider citing this publication when using this code.
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

import os
import json

import tensorflow as tf

from helpers.nn_parametrization import load_nn_params
from helpers.globals import get_prefix, set_activation_function
from helpers.csv_helpers import import_csv

def extract_kpis(create_fun, appl_path, mu_factor,  post_extract_callout):
    """
    Extract key performance indicators for a posteriori error estimation. KPIs are then stored in a KPI file to be persistently accessible for other evaluation steps 

    :param function create_fun: factory function which creates PINN 
    :param string appl_path: path to application main directory. Relative to this, output data will be stored
    :param float mu_factor: smoothening parameter for creating delta from deviation R relative to deltamean
    :param function post_extract_callout: callout to be called after parameter extraction
    """
    # set params for determinism
    tf.config.threading.set_inter_op_parallelism_threads(1) 
    tf.config.threading.set_intra_op_parallelism_threads(1) 
    tf.config.set_soft_device_placement(True)

    # Load parameters
    input_dim, output_dim, N_layer, N_neurons, lb, ub, af = load_nn_params(os.path.join(appl_path, 'config_nn.json'))
    if af is not None:
        set_activation_function(af)

    # PINN initialization
    pinn = create_fun([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)

    # PINN parametrization by stored weights
    weights_path = os.path.join(appl_path, 'output_data', get_prefix() +'weights')
    pinn.load_weights(weights_path)
    X_phys = tf.convert_to_tensor(import_csv(os.path.join(weights_path, "collocation_points.csv")))

    # divide data into time and not time to only watch time variables
    t = X_phys[:,0:1]
    nott = X_phys[:, 1:X_phys.shape[1]]

    # determine mu
    val = tf.squeeze(pinn.f_model(X_phys)**2)
    if tf.rank(val).numpy() >1:
        reducesumneeded = True
        deltaabs =  tf.reduce_sum( val , axis=1)
    else: 
        reducesumneeded = False
        deltaabs = val
    
    if tf.rank(deltaabs) > 1:
        delta_mean = tf.reduce_mean(deltaabs[:, 0])**0.5
    else:
        delta_mean = tf.reduce_mean(deltaabs)**0.5
    mu = tf.cast(tf.fill((X_phys.shape[0],1), delta_mean*mu_factor) , 'float64')

    # determine K 
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            delta = (pinn.f_model(tf.concat([t, nott], axis=1)))**2
            if reducesumneeded:
                delta = tf.reduce_sum(delta, axis=1)
            loss = tf.math.exp(-pinn.get_Lf()*t)*(delta + mu**2)**(0.5)
        dtloss = tape1.gradient(loss, t)
    dtdtloss = tape2.gradient(dtloss, t)

    # export data to kpi json
    data = {}
    data['K'] = tf.reduce_max(tf.abs(dtdtloss)).numpy()
    data['mu'] = delta_mean.numpy()*mu_factor
    data['L_f'] = pinn.get_Lf()
    data['delta_mean'] = delta_mean.numpy()
    
    out_file = open( os.path.join(appl_path, 'output_data', get_prefix()+'kpis.json'), "w")
    json.dump(data, out_file, indent=4)

    if post_extract_callout is not None:
        post_extract_callout(X_phys, deltaabs**0.5, loss, dtloss, dtdtloss,os.path.join(appl_path, 'output_data') )

    del tape1
    del tape2