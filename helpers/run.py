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
import logging

import tensorflow as tf
import numpy as np

from helpers.nn_parametrization import load_ee_params, load_nn_params
from helpers.globals import get_prefix, set_activation_function
from helpers.csv_helpers import export_csv

def compute_integral_trpz(Y, dx):
    """
    Computes integral over Y using trapezoidal rule and step size dx

    :param np.array Y : values to integrate numerically
    :param float dx : step size in numerical integration
    """
    integralvalue = dx/2.0*(np.sum(Y[0:(len(Y)-1)])+np.sum(Y[1:len(Y)]))
    return integralvalue

def Compute_Error(X_data, pinn, K, mu, Lf, deltamean, epsilon, ndim) :
    """
    Function to determine error for input data X_data

    :param array X_data: input data for PINN
    :param PINN pinn: PINN under investigation
    :param float K: key parameter for using trapezoidal rule and estimating the number of required subintervals
    :param float mu: smoothening parameter for creating delta from deviation R
    :param float Lf: Lipschitz constant or spectral abscissa of system under investigation
    :param float deltamean: a priori determined average deviation in ODE/PDE
    :param float epsilon: contribution the error of the numerical integration may give to the overall a posteriori error
    :param int ndim: dimensions of input data  
    """
    # initialize variables for error and number of support points
    E_pred = np.zeros((X_data.shape[0], 2)) 
    N_SP = np.repeat(0, X_data.shape[0], axis=0)

    # compute target value and error for all times
    for x_index in range(X_data.shape[0]):
        # get current item
        x_item = np.reshape(X_data[x_index], (1, X_data.shape[1]))   

        # predict value at time 0 and compare to input values to get r0
        t = x_item[0,0]
        x_item[0,0] = 0
        r0 = np.sqrt(np.sum((pinn.predict(x_item)[0] - x_item[0, -ndim:])**2))
        x_item[0,0] = t

        # compute predicted machine learning error and number of required support points
        E_ML = np.exp(Lf * x_item[0,0])*(r0 + (1-np.exp(-x_item[0,0]*Lf))*deltamean/Lf)
        N_SP[x_index] = np.ceil(np.sqrt(K*x_item[0,0]**3 / (12*E_ML*epsilon))).astype(int)

        # compute prediction of support points
        T_test = np.transpose(np.array([np.linspace(0,x_item[0,0],2*(N_SP[x_index]+1))]))
        X_test = np.repeat(x_item, T_test.shape[0], axis=0)
        X_test[:,0]  = T_test[:,0]
        _, F_pred = pinn.predict(X_test)
    
        # compute integral for error
        targetfun = (np.sqrt(np.reshape(np.sum(F_pred**2, axis=1),(F_pred.shape[0],1)) + np.full((F_pred.shape[0],1), mu, dtype="float64")**2) * np.exp(-Lf*T_test))
        I_1 = compute_integral_trpz(targetfun, T_test[1]-T_test[0]) 
        if x_item[0,0] > 0:
            I_2 = compute_integral_trpz(targetfun[0::2], T_test[2]-T_test[0]) 
        # determine error
        E_pred[x_index, 0] = np.exp(Lf*x_item[0,0])*(r0)
        if x_item[0,0] == 0:
            E_pred[x_index, 1] = 0
        else:
            E_pred[x_index, 1] = np.exp(Lf*x_item[0,0])*(I_1 + 0.75*np.absolute(I_1-I_2))

        if x_index % 100 == 0:
            logging.info(f'Predicted error for index {x_index}: {E_pred[x_index]}')
    return E_pred, N_SP

def eval_pinn(create_fun, load_fun, input_file, appl_path, epsilon, DISABLE_ERROR_ESTIMATION, callout=None):
    """
    Basic function for running a PINN with input data

    :param function create_fun: factory function which creates PINN 
    :param function load_fun: function which loads or creates data points based on load_param
    :param string input_file: path to input data 
    :param string appl_path: path to application main directory. Relative to this, output data will be stored
    :param float epsilon: fraction of error the error introduced by numerical integration may have
    :param boolean DISABLE_ERROR_ESTIMATION: determines if a posteriori error estimation is executed or if the PINN is evaluated only.
    :param function callout: callout to be called after evaluating the PINN    
    """
    # Load parameters
    input_dim, output_dim, N_layer, N_neurons, lb, ub, af = load_nn_params(os.path.join(appl_path,'config_nn.json'))
    if af is not None:
        set_activation_function(af)
    K, mu, Lf, deltamean = load_ee_params(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"))

    # load input values
    if input_file is None:
        logging.warning(f'No input file is given. Load_fun called with None')
        _, X_data, _ = load_fun(None)
    else:
        _, X_data, _ = load_fun(os.path.join(appl_path, "input_data", input_file))

    # PINN initialization + parametrization by stored weights
    pinn = create_fun([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)
    weights_path = os.path.join(appl_path, 'output_data', get_prefix() + 'weights')
    pinn.load_weights(weights_path)

    # run pinn
    Y_pred, _ = pinn.predict(X_data)

    # format export data and column headers
    if not DISABLE_ERROR_ESTIMATION:
        E_pred, N_SP_pred = Compute_Error(X_data, pinn, K, mu, Lf, deltamean, epsilon, output_dim)

    # export data to csv to reuse results
    if input_file is not None:
        export_data = np.concatenate([X_data, Y_pred], axis=1)

        colhdsx = np.core.defchararray.add(np.core.defchararray.add( np.repeat("X_data[", X_data.shape[1], axis=0), np.linspace(0, X_data.shape[1], X_data.shape[1]).astype('int').astype('str')) , np.repeat("]",X_data.shape[1],axis=0))
        colhdsy =  np.core.defchararray.add(np.core.defchararray.add( np.repeat("Y_pred[", Y_pred.shape[1], axis=0), np.linspace(0, Y_pred.shape[1], Y_pred.shape[1]).astype('int').astype('str')) , np.repeat("]",Y_pred.shape[1], axis=0))
        colhds = np.concatenate([colhdsx, colhdsy], axis = 0)
        
        # determine error
        if not DISABLE_ERROR_ESTIMATION:
            export_data = np.concatenate([export_data, E_pred, np.reshape(N_SP_pred, (N_SP_pred.shape[0],1))], axis = 1)
            colhdse = np.core.defchararray.add(np.core.defchararray.add( np.repeat("E_pred[", 2, axis=0), np.linspace(0, 2, 2).astype('int').astype('str')) , np.repeat("]",2, axis=0))
            colhdsn = np.core.defchararray.add(np.core.defchararray.add( np.repeat("N_SP[", 1, axis=0), np.linspace(0, 1, 1).astype('int').astype('str')) , np.repeat("]",1, axis=0))
            colhds = np.concatenate([colhds, colhdse, colhdsn], axis = 0)

        export_csv(export_data, os.path.join(appl_path, 'output_data',"run_" + get_prefix() + input_file), columnheaders=colhds)

    # call post_run_callout
    if callout is not None: 
        callout(os.path.join(appl_path, 'output_data', 'run_'+get_prefix()+'figures'), X_data, Y_pred, E_pred, N_SP_pred)

