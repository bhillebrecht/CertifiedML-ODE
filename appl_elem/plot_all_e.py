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

import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helpers"))

from plotting import new_fig, save_fig
from csv_helpers import import_csv

def plot_all_e():
    egelu = import_csv(os.path.join("appl_elem", "output_data", "gelu_figures", "delta.csv"))
    egelu = egelu[egelu[:,0].argsort()]

    etanh = import_csv(os.path.join("appl_elem", "output_data", "tanh_figures", "delta.csv"))
    etanh = etanh[etanh[:,0].argsort()]

    esilu = import_csv(os.path.join("appl_elem", "output_data", "silu_figures", "delta.csv"))
    esilu = esilu[esilu[:,0].argsort()]
    
    esoftmax = import_csv(os.path.join("appl_elem", "output_data", "softmax_figures", "delta.csv"))
    esoftmax = esoftmax[esoftmax[:,0].argsort()]

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.set( ylabel=r'$\| \mathcal{R}(t) \|$')
    ax.set( xlabel="Time $t$")

    ax.grid(True, which='both')
    ax.set(xlim=[np.min(egelu[:,0]), np.max(egelu[:,0])])

    ax.plot(egelu[:,0], egelu[:,1], linewidth=2,c=colors[0], label="gelu")
    ax.plot(esilu[:,0], esilu[:,1], linewidth=2,c=colors[1], linestyle="--", label="silu")
    ax.plot(esoftmax[:,0], esoftmax[:,1], linewidth=2,c=colors[2], label="softmax")
    ax.plot(etanh[:,0], etanh[:,1], linewidth=2,c=colors[3], linestyle=":", label="tanh")

    ax.legend(loc='best')
    fig.tight_layout()

    save_fig(fig, "compare_e", os.path.join("appl_elem", "output_data","comparison"))
    plt.show()

if __name__ == "__main__":
    plot_all_e()