# BI Analyst and Data Scientist
### **Key Skills: Power Platform, Python, SQL, R**

## Education
- Apprenticeship: L4 Data Analyst, Distinction 
- MSC (Hons): Ocean Science, University of Liverpool, 1st Class
- BSC (Hons): Oceanography, University of Liverpool, 1st Class

## Work Experience
- Advanced Analytics Analyst: Financial Institute
- Data Analyst: Financial Institute
- Data Analyst: Medical Company
- Lab Supervisor: Medical Company

## Projects on Github:
- Northern Ireland Phytoplankton Analysis

# Northern Ireland Phytoplankton Analysis
*Analysis of the Algal Concentrations over time from fisheries around Northern Ireland*
*Built within Jupyter Notebooks using Python 3.13.7*
## Table of Contents
- [Import Packages](#Import-Packages)
- [How to Customize Markdown files?](#how-to-customize-markdown-files)
- [How to Create New Repository?](#how-to-create-new-repository)

# Import Packages
```
## Import relevant packages
import pandas as pd
from pandas.plotting import autocorrelation_plot
import numpy as np
import re
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
