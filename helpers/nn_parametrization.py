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

import json
import numpy as np

def load_training_params(filepath):
    """
    Loads parameters to configure the training from the json file provided via param filepath

    returns
    - epochs: number of epochs used for training
    - n_phys: number of collocation points for the PINN
    """

    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    opt = None
    try: 
        opt = str(data['optimizer'])
    except KeyError as e:
        opt = None    

    return int(data['epochs']), int(data['n_phys']), opt

def load_nn_params(filepath):
    """
    Loads parameters for the neural network from the json file provided via param filepath.

    returns 
    - input_dim: number of input nodes
    - output_dim: number of output nodes
    - num_layers: number of layers
    - num_neurons: number of nodes per layer
    - lower_bound: lower bounds on input node values
    - upper_bound: upper bounds on input node values
    - af (None): if set in json, the value for the activation function
    """

    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    lb = np.array(data['lower_bound'])
    ub = np.array(data['upper_bound'])

    af = None
    try: 
        af = str(data['activation_function'])
    except KeyError as e:
        af = None

    return int(data['input_dim']), int(data['output_dim']), int(data['num_layers']), int(data['num_neurons']), lb.astype(np.float), ub.astype(np.float), af

def load_ee_params(filepath):
    """
    Loads parameters for a posteriori error estimation for the PINN from the json file provided via param filepath.

    returns 
    - K: as used for trapezoidal rule
    - mu: smoothing parameter for delta function
    - L_f: Lipschitz constant or spectral abscissa
    - delta_mean: average deviation of approximated ODE/PDE from target ODE/PDE
    """
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return float(data['K']), float(data['mu']), float(data['L_f']), float(data['delta_mean'])

def has_param(filepath, param_name):
    """
    Checks if keyword param_name exists in json file

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    try: 
        data[param_name] 
    except KeyError as e:
        return False

    return True

def get_param_as_float(filepath, param_name):
    """
    Extracts parameter as float from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return float( data[param_name] )

def get_param_as_int(filepath, param_name):
    """
    Extracts parameter as integer from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """    
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return int( data[param_name] )

def get_param_as_array(filepath, param_name):
    """
    Extracts parameter as array from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """    
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return np.array( data[param_name] )

def get_param_as_string(filepath, param_name):
    """
    Extracts parameter as string from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """    
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return str( data[param_name] )

def get_param_as_boolean(filepath, param_name):
    """
    Extracts parameter as boolean from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """        
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return bool( (data[param_name] == "True") or (data[param_name] == "true") ) 