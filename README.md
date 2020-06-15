# Brainstem_V2a_V1_project

Pipeline for behavior and calcium imaging analysis from experiments performed on head-embedded/Tail-free ZebraFish.
Written in Python.

IDE used is [PyCharm-Community](https://www.jetbrains.com/pycharm/)

## Requirement

It is strongly recommended to use the Python from [conda](https://anaconda.org/anaconda/conda) as a Project Interpreter. 
Most of the packages used in this project are already installed in this Python version. 

### Libraries required

Please set up your interpreter to load following packages (each of them are already available if you are using Python from conda env). 


* [numpy](https://www.numpy.org/), [pandas](https://pandas.pydata.org/) and [scipy](https://www.scipy.org/scipylib/index.html) for object management and data analysis.
* [plotly](https://plot.ly/python/) and [matplotlib](https://matplotlib.org/) as plotting libraries.

You can also check out the documentation following the links.

## Overall organisation

### scripts

Is a python directory with python scripts, will perform the final action, they are the one to run.

### utils

Is a python directory with different python scritps used as libraries.   
A library is a list of functions, each of this function is supposed to perform **one** action (in the legendary world of good coding practicing land).  
You will see them being called at the beginning the the python scripts in *scripts* as **from utils.library1 import function1**. 
  
  
Mostly, you have 2 types of librairies:
* Transversal, with functions being used recursively, as *import_data.py* or *plotting*.
* Applied to a specific script, as *functions_ZZ_extraction.py* being called exclusively by the script *extractZZOutput.py*.  

## Scripts description

