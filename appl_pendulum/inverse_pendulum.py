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

import tensorflow as tf
import numpy as np

from base.pinn import PINN
from helpers.csv_helpers import import_csv
from helpers.plotting import plot_absolute_errors, plot_loss, plot_states_1D, new_fig, save_fig
from helpers.globals import get_prefix
from helpers.error_nn import eval_error_nn

import matplotlib.pyplot as plt

def rhs(q, u):
   """
   Computes right hand side of inverse pendulum ODE based on current values of phi,phi dot, x, xdot
   
   fixed parameters of the example are:
   - J : mass moment of inertia = 0.0361 kg/m^2
   - a : distance between center of mass of the pendulum mass and the waggon = 0.42 m
   - m : mass of the pendulum = 0.3553 kg
   - g : gravitational constant = 9.81 m/s^2
   - d : friction coefficient = 0.005 Nms

   formula:
   d/dt phi = phi_dot
   d/dt phi_dot = (m*g*a*sin(phi) - d*phi_dot + m*a*cos(phi)*u)/(J+m*a^2)
   d/dt x = x_dot
   d/dt x_dot = u
   """
   rhs_1 = q[:, 1:2]
   rhs_2 = (1.46390706*tf.math.sin(q[:,0:1]) # ( m g a sin(phi)
      - 0.005*q[:,1:2]                       # - d phi_dot
      + 0.149226* tf.math.cos(q[:,0:1])*u)/0.09877492 # +m a cos(phi) ) / (J+ma^2)
   rhs_3 = q[:, 3:4]
   rhs_4 = u
   
   return tf.stack([rhs_1, rhs_2, rhs_3, rhs_4], axis = 1)

class InversePendulum(PINN):
    """
    Class used to represent the PINN describing the inverse pendulum
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
        self.t, self.u, self.x0 = None, None, None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
      self.t, self.u, self.x0 = self.tensor(X_f[:,0:1]), self.tensor(X_f[:,1:2]), self.tensor(X_f[:,2:6])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the pendulum.

        :return: tf.Tensor: the prediction of the PINN
        """
        # Dt(y) = Ay + Bu

        if X_f is None:
            t, u, x0 = self.t, self.u, self.x0
        else:
            t, u, x0 = self.tensor(X_f[:,0:1]), self.tensor(X_f[:,1:2]), self.tensor(X_f[:,2:6])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            q = self.model(tf.concat([t, u, x0], axis=1))
            q1, q2, q3, q4 = q[:, 0:1], q[:, 1:2],  q[:, 2:3], q[:, 3:4]
        dq_dt = tf.stack([tape.gradient(q1, t), tape.gradient(q2, t), tape.gradient(q3, t), tape.gradient(q4, t)], axis = 1)
        del tape

        f_pred = dq_dt - rhs(q, u)
        return f_pred

    def get_Lf(self):
        """
        returns lipschitz constant of right hand side

        Upper limit on lipschitz constant manually computed by deriving rhs.
        """
        return 27.09030160668053

def load_data(filepath):
   """
   Loads data from csv file specific for the inverse pendulum simulation
   """
   data = import_csv(filepath)
   x_data = data[:, 0:6]
   y_data = None
   if data.shape[1]>6:
       y_data = data[:, 6:10]
   n_data = data.shape[0]

   return n_data, x_data, y_data

def create_pinn(nn_params, lb, ub):
   return InversePendulum(nn_params, lb, ub)

def post_train_callout(pinn, output_directory):
    return

def post_extract_callout(input, e, delta, deltadot, deltadotdot, output_directory):
    plot_loss(input[:,0:1].numpy(), "e",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$e(t)$", e.numpy()) 
    plot_loss(input[:,0:1].numpy(), "delta",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$\delta (t)$", delta[:,0:1].numpy()) 
    plot_loss(input[:,0:1].numpy(), "deltadotdot",  os.path.join(output_directory, get_prefix()+"figures"), "Time $t$", "$\ddot{\delta}(t)$", deltadotdot[:,0:1].numpy()) 
   
    return

def det_reference(X_data, dt):
    # Define parameters
    factor = 50
    h = dt/factor

    Y_ref = np.zeros(((X_data.shape[0])*factor-1, 4))
    Y_ref[0] = X_data[0, 2:6]
    for index in range(0, X_data.shape[0]-2):
        Y_ref[factor*index] = X_data[index, 2:6]
        for i in range(0, factor):
            Y_ref[index*factor+1+i,:] = Y_ref[index*factor+i,:] + h*np.squeeze(rhs(Y_ref[index*factor+i:(index*factor+i+1),:], X_data[index:index+1, 1:2]))
    return Y_ref, factor

def post_eval_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None ):
    # input may be subdivided in multiple, equally sized repetition arrays with the same input except for time
    if (False):
        t = X_data[:,0]
        repfactor = 1
        dt = t[1]-t[0]
        tmax = dt*t.shape[0]
        for i in range(1, t.shape[0]):
            if t[i] == 0.0:
                repfactor = (i) 
                dt = (i)*dt
                break
        t = np.linspace(t[0], tmax, t.shape[0])

        # compute and export reference solution
        Y_ref, factor = det_reference(X_data[::repfactor], dt)
        Y_ref = Y_ref[::int(factor/repfactor)]
        Y_ref[-1,:] = X_data[-1, 2:6]

        # plot ML solution 
        plot_states_1D(t, Y_ref[:,0], get_prefix() + "phi_dev", outdir, ylabeli=r'$\phi(t)$', legend_ref =r'$\phi(t)$', legend_pred=r'$\hat{\phi}(t)$', Z_pred=Y_pred[:,0])
        plot_states_1D(t, Y_ref[:,1], get_prefix() + "phidot_dev", outdir, ylabeli=r'$\dot{\phi}(t)$', legend_ref =r'$\dot{\phi}(t)$', legend_pred=r'$\dot{\hat{\phi}}(t)$',Z_pred=Y_pred[:,1])
        plot_states_1D(t, Y_ref[:,2], get_prefix() + "s_dev", outdir, ylabeli=r'$s(t)$', legend_ref =r'$s(t)$', legend_pred=r'$\hat{s}(t)$',Z_pred=Y_pred[:,2])
        plot_states_1D(t, Y_ref[:,3], get_prefix() + "sdot_dev", outdir,ylabeli=r'$\dot{s}(t)$', legend_ref =r'$\dot{s}(t)$', legend_pred=r'$\dot{\hat{s}}(t)$', Z_pred=Y_pred[:,3])


        linewidth = 2
        colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
        fig = new_fig()
        ax = fig.add_subplot(2, 2, 1)
        ax.set(ylabel=r'$\phi(t)$')
        ax.set(xlim=[np.min(t), np.max(t)])
        ax.plot(t,  Y_ref[:,0], linewidth=linewidth, label=r'$\phi(t)$', c=colors[0])
        ax.plot(t, Y_pred[:,0], linestyle=':', linewidth=linewidth, label=r'$\hat{\phi}(t)$',
                        c=colors[1])
        ax.grid('on')
        ax.legend(loc='best')

        ax = fig.add_subplot(2, 2, 2)
        ax.set( ylabel=r'$\dot{\phi}(t)$')
        ax.set(xlim=[np.min(t), np.max(t)])
        ax.plot(t,  Y_ref[:,1], linewidth=linewidth, label=r'$\dot{\phi(t)}$', c=colors[0])
        ax.plot(t, Y_pred[:,1], linestyle=':', linewidth=linewidth, label=r'$\dot{\hat{\phi}}(t)$',
                        c=colors[1])
        ax.grid('on')
        ax.legend(loc='best')

        ax = fig.add_subplot(2, 2, 3)
        ax.set(xlabel='Time $t$', ylabel=r'$s(t)$')
        ax.set(xlim=[np.min(t), np.max(t)])
        ax.plot(t,  Y_ref[:,2], linewidth=linewidth, label=r'$s(t)$', c=colors[0])
        ax.plot(t, Y_pred[:,2], linestyle=':', linewidth=linewidth, label=r'${\hat{s}}(t)$',
                        c=colors[1])
        ax.grid('on')
        ax.legend(loc='best')

        ax = fig.add_subplot(2, 2, 4)
        ax.set(xlabel='Time $t$', ylabel=r'$\dot{s}(t)$')
        ax.set(xlim=[np.min(t), np.max(t)])
        ax.plot(t,  Y_ref[:,3], linewidth=linewidth, label=r'$\dot{s}(t)$', c=colors[0])
        ax.plot(t, Y_pred[:,3], linestyle=':', linewidth=linewidth, label=r'$\dot{\hat{s}}(t)$',
                        c=colors[1])
        ax.grid('on')
        ax.legend(loc='best')
        fig.tight_layout()

        save_fig(fig, "all_components_pendulum", outdir)
        plt.show()

        E_abs = np.sqrt(np.sum((Y_pred - Y_ref)**2, axis=1))
        plot_absolute_errors(t, E_pred, "abserr", outdir, Y_ref, Y_pred )
        plot_absolute_errors(t[0:125], E_pred[0:125,:], "abserr_first125", outdir, Y_ref[0:125,:], Y_pred[0:125] )

    t = X_data[:,0]
    dt = t[1]-t[0]
    tmax = dt*t.shape[0]
    t = np.linspace(t[0], tmax, t.shape[0])   

    E_NN = eval_error_nn(os.path.join(outdir, '..','..'), 'config_error_nn.json', X_data)
    
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

