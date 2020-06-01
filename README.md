# RSN_project

Pipeline for data extraction and analysis oof calcium imaging experiments performed on head-embedded/Tail-free ZebraFish.
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


### extract_raw_data_fish.py

From the software ZebraZoom outputs, create datasets with raw data and some kinematic parameters calculated.
  
### Functions_ZZ_extraction.py 

*Library* of functions used to extract and calculate parameters assigned in DataFrames of **Extract_raw_data_fish.py
  
### behavior_analysis.py

Correct raw data and assign category to bouts.
  
### Calcium_analysis.py

DFF calculation, comparison with behavior classified by type. 
  
### Functions_analysis.py

*Library* of functions used both for **Behavior_analysis.py** and **DFF_analysis_finale.py**

