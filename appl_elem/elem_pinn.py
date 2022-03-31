###################################################################################################
# (c) 2022 Birgit Hillebrecht
# 
# This code has been developed as part of [TBD insert link to pub]
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

from matplotlib import pyplot as plt
from helpers.csv_helpers import export_csv, import_csv

import numpy as np
import tensorflow as tf

from base.pinn import PINN
from helpers.error_nn import eval_error_nn
from helpers.globals import get_prefix
from helpers.nn_parametrization import get_param_as_boolean, get_param_as_float, get_param_as_int, has_param
from helpers.plotting import new_fig, plot_states_1D, plot_loss, plot_trustzone, plot_absolute_errors, save_fig

class ElemPINN(PINN):
    """
    Class used to represent the PINN describing an elementary ODE of a decaying exponential function.
    """

    def __init__(self, layers, lb, ub, X_f=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub)
        self.t = None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def override_loss(self, w_smallt, w_larget):
        self.w_smallt = w_smallt
        self.w_larget = w_larget
        self.loss_object = self.weighted_loss

    def set_collocation_points(self, X_f):
        self.t = self.tensor(X_f[:, 0:1])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual Physics Informed Neural Network for the evaluation of the linear ODE Dq + 2q = 0

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
        else:
            t = self.tensor(X_f[:, 0:1])

        with tf.GradientTape() as tape:
            tape.watch(t)
            q = self.model(t, training=False)
            dq_dt = tape.gradient(q, t)

        f_pred = dq_dt+2*q

        return f_pred

    @tf.function
    def weighted_f_model(self):
        """
        This function can be used to introduce a weighted loss without modifying the a posteriori used f_model function. 
        """
        t = self.t
        f = self.f_model()
        f_pred = self.w_smallt*(tf.math.exp(-self.get_Lf()*t)+self.w_larget*tf.math.exp(self.get_Lf()*t))*f

        return f_pred

    def get_Lf(self):
        L_f = -2.0
        return L_f

    def weighted_loss(self, y, y_pred):
        w_data = 1
        w_phys = 1

        f_pred = self.weighted_f_model()
        L_data = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), axis=1))
        L_phys = tf.reduce_mean(tf.reduce_sum(tf.square(f_pred), axis=1))
        
        L = w_data * L_data + w_phys * L_phys

        return L

def create_pinn(nn_params, lb, ub):
    pinn = ElemPINN(nn_params, lb, ub)

    config_filepath = os.path.join(os.path.dirname(__file__), "config_training.json")
    # if configured, override loss function by weighted loss function
    if has_param(config_filepath, 'override_loss'):
        if get_param_as_boolean(config_filepath, 'override_loss'):
            logging.info("Loss will be set to weighted loss")
            if has_param(config_filepath, 'w_larget') and has_param(config_filepath, 'w_smallt'):
                pinn.override_loss(get_param_as_float(config_filepath, 'w_larget'), get_param_as_float(config_filepath, 'w_smallt'))
                logging.info("Parameters are: " + str(get_param_as_float(config_filepath, 'w_smallt')) + " (small t), " + str(get_param_as_float(config_filepath, 'w_larget'))+ " (large t)")
            else :
                logging.warning("Weights for weighted loss are not validly given according to override loss configuration")
    return pinn

def load_data(loadParam):
    if loadParam is None:
        X = import_csv(os.path.join("appl_elem", "input_data", "check_data.csv"))
        return 1, X[0:1,0:1], X[0:1, 1:2]    
    else:
        X = import_csv(loadParam)
        return X.shape[0], X, None

def post_train_callout(pinn, output_directory):
    # in post training evaluate test sequence and plot
    X_test = np.transpose(np.array([np.linspace(0,2,50)]))
    Y_test = 2*np.exp(-2*X_test)
    Y_pred, F_pred = pinn.predict(X_test)

    plot_states_1D(X_test, Y_test, "data_elempinn", os.path.join(output_directory, get_prefix()+"figures"), Z_pred=Y_pred)
    
    # plot absolute error
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ ', ylabel=r'Absolute Error')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set(xlim=[np.min(X_test), np.max(X_test)])

    abs_errors = np.abs(Y_test - Y_pred)
    ax.plot(X_test, abs_errors, linewidth=2)

    ax.legend(loc='best')
    save_fig(fig, "absolueerr_elempinn", os.path.join(output_directory, get_prefix() +"figures"))
    fig.tight_layout()
    plt.show()
    
    return

def post_extract_callout(input, e, delta, deltadot, deltadotdot, output_directory):
    plot_loss(input.numpy(), "e",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$e(t)$", e.numpy()) 
    plot_loss(input.numpy(), "delta",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$\delta (t)$", delta.numpy()) 
    plot_loss(input.numpy(), "deltadot",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$\dot{\delta} (t)$", deltadot.numpy()) 
    plot_loss(input.numpy(), "deltadotdot",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$\ddot{\delta}(t)$", deltadotdot.numpy()) 
    
    #export data for comparison plot
    data = np.concatenate((input.numpy(),np.reshape(e.numpy(), input.numpy().shape)), axis=1)
    export_csv(data, os.path.join(output_directory, get_prefix()+"figures","delta.csv") )
    return

def post_eval_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None ):
    plot_trustzone(X_data[:,0:1], Y_pred, E_pred[:,0]+E_pred[:,1], 2*np.exp(-2*X_data[:,0:1]), get_prefix()+"trustzone", outdir)
    plot_absolute_errors(X_data[:,0:1], E_pred, get_prefix() + "abserr_vs_prederr", outdir, Z_pred=Y_pred,  Z_ref=  2*np.exp(-2*X_data[:,0:1]))
    plot_states_1D(X_data[:,0], N_SP_pred,  get_prefix()+"N_SP", outdir)
    
    t = X_data[:,0]
    E_NN = eval_error_nn(os.path.join(outdir, '..','..'), 'config_error_nn.json', t)
    
    colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:blue']
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ ', ylabel=r'Error')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set(xlim=[np.min(t), np.max(t)])

    ax.plot(t, E_pred[:,0]+E_pred[:,1], linewidth=2, color= colors[0], label=r'$E_\mathrm{Init} + E_\mathrm{PI}$')
    ax.plot(t, E_NN[:], linewidth=2, linestyle="--",  color= colors[3], label=r'$E_\mathrm{NN}$')

    ax.legend(loc='best')
    save_fig(fig, "error_NN", outdir)
    fig.tight_layout()
    plt.show()

    return

