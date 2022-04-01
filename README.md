# Certified Physics Informed Neural Network 

Before using the here published code, be aware that the code is published under MIT common license, as found in the LICENSE file enclosed. To use this work in academic publications, we kindly ask to consider citing

  Certified machine learning: A posteriori error estimation for physics-informed neural networks
  https://doi.org/10.48550/arXiv.2203.17055

The code provides a standard mean to use certified PINNs for your system, which is goverened by an ODE or PDE. 

## System Requirements

Install python 3.9.x, tensorflow 2.7.0 is still incompatible to python 3.10.x.
Use pip to install all packages listed in requirements.txt
```
pip install -r requirements.txt
```

## Configuration 

The two examples which were published in the above mentioned application are included as well. The here published code has two means to be configured and controlled:
- static parameters of the neural network and the training are set via configuration files
- commands and commandline options help control the neural networks and the current actions applied to those

### Configuration Files

Static configuration of the neural network and the training is done via .json files. One for the neural network in general
- "config_nn.json" with parameters
    - (uint, mandatory) input_dim : dimension of input, equals number of input neurons
    - (uint, mandatory) output_dim : dimension of output, equals number of output neurons
    - (uint, mandatory) num_layers : number of layers between input layer and output layer
    - (uint, mandatory) num_neurons: number of neurons per layer between input and output layer
    - (array, mandatory) lower_bound : array of lower limits on input parameter for collocation point generation and network configuration
    - (array, mandatory) upper_bound : array of lower limits on input parameter for collocation point generation and network configuration
    - (string, optional) activation_function: Activation function used for the neural network, valid values are tanh, silu, gelu and softmax

and one for the training process
- "config_training.json" with mandatory parameters
    - (uint, mandatory) epochs : number of used training epochs
    - (uint, mandatory) n_phys : number of used/generated collocation points for training
    - (string, optional) optimizer: Optimizer used during training, valid values are adam and lbfgs

### Command Line Configuration

Using commands, subcommands and command line options, you can control the framework. To find a full description of available options and commands, use

```
python .\certified_pinn.py --help
```

The central control unit has four subcommands, which depend on the results of the previous one
1. train
2. extract
3. either 
    1. run or
    2. train-error-net

In consequence, extract can not be called without training the NN before and run and train-error-net can not be called without extracting key parameters before using the command extract.

To find close information on the options of the commands use e.g. for "train"
```
python .\certified_pinn.py -t elem train --help
```

Let three example commands be given as easy way to get started with actually running the code: 
1. training the exponential decay application
```
python .\certified_pinn.py -t elem train 
```
2. training the exponential decay application
```
python .\certified_pinn.py -t elem extract 
```
3. training the exponential decay application
```
python .\certified_pinn.py -t elem run -i check_data.csv
```

## Example Applications

For a detailled description of the two example applications, refer to **TODO: Insert arxiv link**.

### 1D : Exponential Decay

The exponential decay example has two peculiarities in implementation
- it uses additional configuration options in "config_training.json" to realize a weighted training, which favors either optimization for small times or for large times. This is realized completely outside standard files but only in the factory function "create_pinn" in appl_elem/elem_pinn.py.
- the neural network has no dependency on initial conditions, however, for the a posteriori error estimation this information on the initial condition is needed to allow proper evaluation. 

### 4D : Inverted Pendulum

The inverted pendulum is the standard case of the application. Special handling is only included for the run call, since the input data for the run call encompasses multiple intervals to be evaluated and creating one joint time series requires special handling in "post_eval_callout".

An adequate run call would be
```
python .\certified_pinn.py -t pendulum run -i repeat_input.csv
```

## Integrating Your System

To integrate your own use case into this framework, use the provided template files in the "_template_appl" directory. Implement all functions in this file even though they might contain some example code - even though the example code might by chance fit your application, it is recommended to replace the complete function by your own implementation.

For this, remember the following 
- the folder and file structure expected is as follows:
    - your_application_folder
        - input_data
        - output_data
        - config_nn.json
        - config_training.json
        - config_error_nn.json
        - config_error_training.json
        - your_application_main_file.py
- input data for run or train must be given relative to the input_data directory
- make sure that input and output dimensions of your problem are properly reflected in data and configuration files
    - in case you want to avoid dependency of the NN on the initial conditions, refer to the 1D example for a hint on how to handle this.

Then, the following command line configuration can be used to execute your application
```
python .\certified_pinn.py -t user -u \<relative or absolute path to your_application_folder\> train -i \<path to your input data relative to input_data directory\>
```
