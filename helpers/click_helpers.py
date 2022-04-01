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
import sys

import logging
import click
from pathlib import Path
from helpers.error_nn import generate_error_training_data_and_train, train_error_nn
from helpers.run import eval_pinn

from helpers.train import train_pinn
from helpers.extract import extract_kpis
from helpers.globals import set_activation_function, set_optimizer

import importlib

TARGET = None
TARGET_PATH = None

def set_target():
    global TARGET
    global TARGET_PATH
    # select target
    appl_path = os.path.dirname(__file__)
    if TARGET == 'elem':
        TARGET_PATH = os.path.join(os.path.dirname(__file__), '..', 'appl_elem', 'elem_pinn.py')

    elif TARGET == 'pendulum':
        TARGET_PATH = os.path.join(os.path.dirname(__file__), '..', 'appl_pendulum', 'inverse_pendulum.py')

    loader = importlib.machinery.SourceFileLoader(Path(TARGET_PATH).stem, TARGET_PATH)
    spec = importlib.util.spec_from_loader(Path(TARGET_PATH).stem, loader)
    mymodule = importlib.util.module_from_spec(spec)
    loader.exec_module(mymodule)

    create_pinn = mymodule.create_pinn
    load_data = mymodule.load_data
    post_train_callout = mymodule.post_train_callout
    post_extract_callout = mymodule.post_extract_callout
    post_eval_callout = mymodule.post_eval_callout
    appl_path = os.path.dirname(TARGET_PATH)

    return appl_path, create_pinn, load_data, post_train_callout, post_extract_callout, post_eval_callout

@click.group()
@click.option('-t', '--target', type=click.Choice(['elem', 'pendulum', 'user'], case_sensitive=False), required=True, help='Selects target, valid values are elem, pendulum or user')
@click.option('-u', '--user_file', required=False, help='Sets the user file to be imported and used during training/extraction and running the PINN. It must contain a collection of mandatory functions, as defined in _template_appl. The parent directory is used as base directory. It is expected to find both, input and output directories next to it as well as required config files.' )
@click.option('-ll', '--log_level', type=click.Choice(['info', 'warning', 'error']), default="info", help='Sets log level accordingly. Valid values are info, warning, error')
@click.pass_context
def click_helper_select(ctx, target, user_file, log_level):
    # set log level as specified in command (default == info)
    if log_level == 'info' :
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    if target == 'user' and user_file is None:
        logging.error("Base directory must be set in case of a user target")
        sys.exit()
    
    if user_file is not None: 
        global TARGET_PATH
        TARGET_PATH = os.path.abspath(user_file)
        logging.info("Absolute path to user file is \'"+ TARGET_PATH + "\'.")

    # store target to import adequate libs and params in sub commands
    global TARGET 
    TARGET = target

@click_helper_select.command()
@click.option('-dae', '--disable_aposteriori_error', is_flag=True, help="Disables automatic error computation" )
@click.option('-i', '--input_file', required = True, help="Set input file on which the network shall be evaluated")
@click.option('-eps', '--epsilon', default=0.33, required=False, type=float, help="Set fraction of expected error in the machine learning prediction that might be added due to numerical integral computation")
@click.pass_context
def run(ctx, disable_aposteriori_error, input_file, epsilon):
    appl_path, create_fn, load_fn, _, _, pevco = set_target()
    eval_pinn(create_fn, load_fn, input_file, appl_path, epsilon, disable_aposteriori_error, pevco)

@click_helper_select.command()
@click.option('-lw', is_flag=True, help='Load weights from predefined path (output_data/weights). This can only be used if data has been stored previously. Default = false')
@click.option('-nsr',  default=False,  help='Avoid storing results, by default, neither weights nor result parameters are stored')
@click.option('-i', '--input_file', required=False, type=click.STRING, help='Sets input file for training')
@click.pass_context
def train(ctx, lw, nsr, input_file):  
    # set target
    appl_path, create_fn, load_fn, ptco, _, _ = set_target()

    # set store results parameter
    sr = not nsr
    
    # train pinn
    train_pinn(create_fn, load_fn, input_file, lw, sr, appl_path, ptco)

@click_helper_select.command()
@click.option('--mu_factor', type=float, default=0.1, help='Sets the factor to multiply avg deviation of ODE with to smoothen upper limit on deviation.')
@click.pass_context
def extract(ctx, mu_factor):
    appl_path, create_fn, _, _, peco, _= set_target()
    extract_kpis(create_fn, appl_path, mu_factor, peco)

@click_helper_select.command()
@click.option('-i', '--input_file', required = False, help="Set input file on which the network shall be evaluated")
@click.option('-eps', '--epsilon', default=0.33, required=False, type=float, help="Set fraction of expected error in the machine learning prediction that might be added due to numerical integral computation")
@click.option('--reset', is_flag=True, help="If set to true, weights for error NN are reset")
@click.pass_context
def train_error_net(ctx, input_file, epsilon, reset):  
    appl_path, create_fn, _, _, _, _ = set_target()
    print(reset)
    # train pinn
    generate_error_training_data_and_train(appl_path, create_fn, epsilon, input_file, not reset)