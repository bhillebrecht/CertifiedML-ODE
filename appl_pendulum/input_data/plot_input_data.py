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
import pandas
import numpy as np
import matplotlib.pyplot as plt

from helpers.plotting import new_fig

def plot_inputdata():
    df = pandas.read_csv(os.path.join("appl_pendulum", "input_data", "pendel.csv")).to_numpy()
    T = df[:,0]
    phi = df[:,2]

    fig = new_fig()
    ax = fig.add_subplot(1,1,1)

    ax.set(xlabel='Time $t$', ylabel=r'$\phi(t)$')
    ax.set(xlim=[np.min(T), np.max(T)])

    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

    ax.plot(T, phi, linewidth=1.5, label=f'$\phi$', c=colors[2])

    plt.show()
