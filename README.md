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
- Northern Ireland Phytoplankton Analysis in Python

# Northern Ireland Phytoplankton Analysis
*Analysis of the Algal Concentrations over time from fisheries around Northern Ireland*
*Built within Jupyter Notebooks using Python 3.13.7*
## Script table of contents
- [1. Introduction](#introduction)
- [2. Import](#import)
  - [2a. Import Packages](#import-packages)
  - [2b. Connect to Github stored data files](#connect-to-github-stored-data-files)
  - [2c. Append each timeseries file](#append-each-timeseries-file)
- [3.Analysis](#analysis)
  - [3a. Exploratory Analysis](#exploratory-analysis)
  - [3b. Data Transformation](#data-transformation)
  - [3c. Format data for Visual Analysis](#format-data-for-visual-analysis)
  - [3d. Group and Label Data](#group-and-label-data)
  - [3e. Visualise Data](#visualise-data)
  - [3f. Timeseries Plot and Outlier Removal](#timeseries-plot-and-outlier-removal)
  - [3g. Data Transformation for Timeseries Forecasting](#data-transformation-for-timeseries-forecasting)
- [4. Timeseries Forecasting](#timeseries-forecasting)
  - [4a. SARIMA Stationarity Test](#sarima-stationarity-test)
  - [4b. Autocorreletation plots for p,d,q,P,D,Q,s determination](#autocorreletation-plots-for-p,d,q,p,d,q,s-determination)
  - [4c. Hyperparameter Optimisation](#hyperparameter-optimisation)
  - [4d. Best AIC Model](#best-aic-model)
  - [4e. Best MAE Model](#best-mae-model)
  - [4f. Best MSE Model](#best-mse-model)
  - [4g. Best Manual Model](#best-manual-model)
- [5. Overview](#overview)

### **Introduction**
Marine toxins may increase with the effects of climate change (Meng et al., 2024). This analysis will investigate the concentration of deadly toxin producing phytoplankton within seven shellfish farms around Northern Ireland over time using data from the Food Standards Agency. Multiple csv’s have been extracted and appended to produce a timeseries between 2022-2025. Python libraries Pandas, Numpy and Matplotlib have been used to cleanse, transform and analyse the raw incomplete data. Statistical techniques such as data normalisation and timeseries forecasting reveal skewed data towards DSP algae and a strong seasonal trend. The SARIMA model developed accounts for seasonality and stationarity through differencing and hyper-parameter optimisation.
Six csv exports of years 2020-2025 have been taken from the Food Standards Agency depicting toxic phytoplankton concentrations within Northern Ireland fisheries (Northern Ireland Phytoplankton Results). Each export contains 9 columns and between 171-386 rows with 2025 being half year.

Below is detail of the Python script written to understand the questions:
  - *How are toxic algal species distributed throughout Northern Ireland Fisheries?*
  - *How are Northern Ireland Algal concentrations predicted to change over time?*

## Import
Python libraries Pandas within VS Code was used to import the 6 csv files and convert to a data frame. Python was used due to the enhanced capability for modelling with accepted packages and the focus of data analysis in this project over a tool such as Power BI which is focused towards visuals. The head of 2025 was viewed to ensure workability and initial data quality audit. This revealed 2025 data included 2024 sampling dates. To ensure data was representative of the year period and dates outside of the year period were filtered.

### **Import Packages**
Import relevant packages for analysis of Algal Concentration
```Python
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
### **Connect to Github stored data files**
Import relevant packages for analysis of Algal Concentration
```Python
# Connect to Github repository for NI Phytoplankton Data 2020-2025
phytodf2025 = pd.read_csv('https://raw.githubusercontent.com/ClaudiaAlice/L6-Data-Science/refs/heads/main/2025_NIPhytoplanktonData.csv')
phytodf2024 = pd.read_csv('https://raw.githubusercontent.com/ClaudiaAlice/L6-Data-Science/refs/heads/main/2024_NIPhytoplanktonData.csv')
phytodf2023 = pd.read_csv('https://raw.githubusercontent.com/ClaudiaAlice/L6-Data-Science/refs/heads/main/2023_NIPhytoplanktonData.csv')
phytodf2022 = pd.read_csv('https://raw.githubusercontent.com/ClaudiaAlice/L6-Data-Science/refs/heads/main/2022_NIPhytoplanktonData.csv')
phytodf2021 = pd.read_csv('https://raw.githubusercontent.com/ClaudiaAlice/L6-Data-Science/refs/heads/main/2021_NIPhytoplanktonData.csv')
phytodf2020 = pd.read_csv('https://raw.githubusercontent.com/ClaudiaAlice/L6-Data-Science/refs/heads/main/2020_NIPhytoplanktonData.csv')

# View head of dataframe
phytodf2025.head()
```

### **Append each timeseries file**

Ensure each export relates only to relevant date, ie 2025 export only contains 2025 data.
```Python
# Before appending each data frame the representative data must be correct, i.e. 2024 survey only contains 2024 results
# 2025
phytodf2025['DateOfSampling'] = pd.to_datetime(phytodf2025['DateOfSampling'])#, format='%d/%m/%Y', dayfirst=True)
phytodf2025 = phytodf2025.loc[(phytodf2025['DateOfSampling'] >= '2025-01-01 00:00:00') & (phytodf2025['DateOfSampling'] < '2026-01-01 00:00:00')]
phytodf2025['Snapshot Date'] = '2025'

# Test to see if filter works 
max_date = phytodf2025['DateOfSampling'].max()
print(max_date)
min_date = phytodf2025['DateOfSampling'].min()
print(min_date)

# Apply to rest of datasets
## 2024
phytodf2024['DateOfSampling'] = pd.to_datetime(phytodf2024['DateOfSampling'], format='%d/%m/%Y', dayfirst=True)
phytodf2024 = phytodf2024.loc[(phytodf2024['DateOfSampling'] >= '2024-01-01 00:00:00') & (phytodf2024['DateOfSampling'] < '2025-01-01 00:00:00')]
phytodf2024['Snapshot Date'] = '2024'

## 2023
phytodf2023['DateOfSampling'] = pd.to_datetime(phytodf2023['DateOfSampling'], format='%d/%m/%Y', dayfirst=True)
phytodf2023 = phytodf2023.loc[(phytodf2023['DateOfSampling'] >= '2023-01-01 00:00:00') & (phytodf2023['DateOfSampling'] < '2024-01-01 00:00:00')]
phytodf2023['Snapshot Date'] = '2023'

## 2022
phytodf2022['DateOfSampling'] = pd.to_datetime(phytodf2022['DateOfSampling'], format='%d/%m/%Y', dayfirst=True)
phytodf2022 = phytodf2022.loc[(phytodf2022['DateOfSampling'] >= '2022-01-01 00:00:00') & (phytodf2022['DateOfSampling'] < '2023-01-01 00:00:00')]
phytodf2022['Snapshot Date'] = '2022'

## 2021
phytodf2021['DateOfSampling'] = pd.to_datetime(phytodf2021['DateOfSampling'], format='%d/%m/%Y', dayfirst=True)
phytodf2021 = phytodf2021.loc[(phytodf2021['DateOfSampling'] >= '2021-01-01 00:00:00') & (phytodf2021['DateOfSampling'] < '2022-01-01 00:00:00')]
phytodf2021['Snapshot Date'] = '2021'

## 2020
phytodf2020['DateOfSampling'] = pd.to_datetime(phytodf2020['DateOfSampling'], format='%d/%m/%Y', dayfirst=True)
phytodf2020 = phytodf2020.loc[(phytodf2020['DateOfSampling'] >= '2020-01-01 00:00:00') & (phytodf2020['DateOfSampling'] < '2021-01-01 00:00:00')]
phytodf2020['Snapshot Date'] = '2020'

# Append results
phytodf = pd.concat([phytodf2020, phytodf2021, phytodf2022, phytodf2023, phytodf2024, phytodf2025], ignore_index=True)

print(phytodf2025['Snapshot Date'].describe)
```
## Analysis
Exploratory analysis of the single data frame was conducted revealing mixtures of data types, inconsistent labelling and null handling, skewed sampling site frequencies and redundant columns. Panda functions .describe and .info could not be implemented here due to lack of data types and inconsistent data.

### **Exploratory Analysis**
```Python
print('Data frame size:\n', phytodf.size,
      '\n\nColumn Data Types:\n',phytodf.dtypes,
      '\n\n',phytodf['Lough'].value_counts(),
      '\n Species \n', phytodf['ShellfishSpecies'].value_counts(),

      '\n Status \n', phytodf['Status'].value_counts(),
      '\n DSPAlgae \n', phytodf['DSPAlgae'].value_counts(),
      '\n Comments \n', phytodf['Comments'].value_counts()
)
```
### **Format data for Visual Analysis**
First step in cleaning the data consisted of removing blank rows, removing unnecessary data such as the columns Comments and Other which only appeared in the 2023-2025 datasets, and renaming the columns to more relevant attributes

Labelling and null indicators is not consistent within the data frame. This may be due to multiple years being appended or different data inputters. This creates columns that are mixes of data types, e.g. DSPAlgae containing 112 and No harmful species and also makes filtering difficult. Trimming all strings and replacing values has been used to convert the Algae column text to 0 indicating no species observed, figure 6a. Data types have been assigned to allow for use of REGEX expression when converting all variations of Open in Status column allowing for filtering to open fisheries

Creation of a month/year column to allow for simpler timeseries analysis.
```Python
# Remove blank rows
phytodf.dropna(how='all')

# # Remove 'Comments' and 'Other' Column using indexing
phytodf.drop(phytodf.columns[-1], axis=1, inplace=True)
phytodf.drop(phytodf.columns[-2], axis=1, inplace=True)

# # Rename Columns
phytodf.rename(columns={ 'Lough':'Location', 
                    'RepresentativeMonitoringPoints':'Sampling Point',
                    'ShellfishSpecies':'Species',
                    'DateOfSampling':'Date',
                    'DSPAlgae':'DSP Algae',
                    'ASPAlgae':'ASP Algae',
                    'PSPAlgae':'PSP Algae',
                    'Snapshot Date':'Snapshot Date'
                    }, inplace=True)

# Trim text columns
phytodf = phytodf.astype(str)
phytodf = phytodf.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)

# # Replace Values 
phytodf = phytodf.replace('No harmful species observed', 0)
phytodf = phytodf.replace('NO harmful species observed', 0)
phytodf = phytodf.replace('no harmful species observed', 0)
phytodf = phytodf.replace('nan', 0)
phytodf = phytodf.replace('No harmful species identified', 0)
phytodf = phytodf.replace('No harmful species', 0)
phytodf = phytodf.replace('no harmful species', 0)
phytodf = phytodf.replace('Oysters / Mussels', 'Oysters/Mussels')
phytodf = phytodf.replace('Mussels/ Oysters', 'Oysters/Mussels')
phytodf = phytodf.replace('Carliingford', 'Carlingford')
phytodf = phytodf.replace('Dundrum BAY', 'Dundrum Bay')
phytodf = phytodf.replace('Dundrum BAY', 'Dundrum Bay')

# Assign Data types
phytodf = phytodf.astype({'Location':str, 
             'Sampling Point':str,
             'Species':str,
             'Date':str,
             'Status':str,
             'DSP Algae':int,
             'ASP Algae':int,
             'PSP Algae':int,
             'Snapshot Date':int
             })

# Replace values in Status column - variations of open using regex \b to match entire word
def replace_open(text, word, replacement):
    return re.sub(rf'\b{word}\b', replacement, text, flags=re.IGNORECASE)

phytodf['Status'] = phytodf['Status'].apply(lambda x: replace_open(x, 'open', 'Open'))

# Create month year column
phytodf['Date'] = pd.to_datetime(phytodf['Date'])
phytodf['Date_fday'] = '01'
phytodf['Date_MY'] = phytodf['Date_fday'] + '-' + phytodf['Date'].dt.strftime('%m-%Y')
phytodf['Date_MY'] = pd.to_datetime(phytodf['Date_MY'],dayfirst=True)  

print(
    'ASP Algae \n', phytodf['ASP Algae'].describe(),
    '\n--- \nDSP Algae \n', phytodf['DSP Algae'].describe(),
    '\n--- \nPSP Algae \n', phytodf['PSP Algae'].describe()
    )

phytodf.info()
```
### **Group and Label Data**
Data was labelled using dictionaries and functions to allow for simplified grouping by location and species by mean when plotting visuals.
```Python
# Sum all aglae as a new column
phytodf['All Algae'] = phytodf['PSP Algae'] + phytodf['ASP Algae'] + phytodf['DSP Algae'] 

# split by locations - create a data frame for each
phytodfC = phytodf.loc[phytodf['Location'] == "Carlingford"]
phytodfB = phytodf.loc[phytodf['Location'] == "Belfast"]
phytodfS = phytodf.loc[phytodf['Location'] == "Strangford"]
phytodfLF = phytodf.loc[phytodf['Location'] == "Lough Foyle"]
phytodfDB = phytodf.loc[phytodf['Location'] == "Dundrum Bay"]
phytodfK = phytodf.loc[phytodf['Location'] == "Killough"]
phytodfLL = phytodf.loc[phytodf['Location'] == "Larne Lough"]

# Create a dictionary and define labels
dataframes = {
    'All': phytodf,
    'Carlingford': phytodfC,
    'Belfast': phytodfB,
    'Strangford': phytodfS,
    'Lough Foyle': phytodfLF,
    'Dundrum Bay': phytodfDB,
    'Killough': phytodfK,
    'Larne Lough': phytodfLL
}

# create empty dictionaries
means = {}
median = {}
vcounts = {}
sumAllphyto = {}
sumPSPphyto = {}
sumDSPphyto = {}
sumASPphyto = {}

# load data into dictionaries for use
for location, df in dataframes.items():
    means[location] = np.mean(df['All Algae']) if not np.isnan(np.mean(df['All Algae'])) else 0
    median[location] = np.median(df['All Algae'])
    sumAllphyto[location] = df.groupby('Location')['All Algae'].sum()
    sumPSPphyto[location] = df.groupby('Location')['PSP Algae'].sum()
    sumDSPphyto[location] = df.groupby('Location')['DSP Algae'].sum()
    sumASPphyto[location] = df.groupby('Location')['ASP Algae'].sum()
    vcounts[location] = df['Species'].value_counts()

# Printing data dictionaries
for location in dataframes.keys():
    print('All Algae:')
    print(f'Means \n {location}: {means[location]:.2f}')
    print(f'Medians \n {location}: {median[location]:.2f}')
    print(f'ValueCounts \n {location}: {vcounts[location]}')
    print(f'sum \n {location}: {sumAllphyto[location]}')
    print("===================")
```
### **Visualise distribution**
To determine the distribution of the data within each location the algal concentrations were then normalised and plotted as a percentage. This revealed DSP species consistently account for over 90% of algal concentrations within all areas.

Further analysis of distribution of algal concentrations and location using mean and standard deviation reveals highly distributed data at Belfast, Dundrum Bay and Lough Foyle. The standard deviation at Lough Foyle is very large indicating potential presence of anomalies
```Python
#Plotting mean and standard deviation of Algae by Location and Algal type
xlabels = phytodf['Location'].unique()
yall = phytodf.groupby('Location')['All Algae'].mean()
ypsp = phytodf.groupby('Location')['PSP Algae'].mean()
yasp = phytodf.groupby('Location')['ASP Algae'].mean()
ydsp = phytodf.groupby('Location')['DSP Algae'].mean()
sasp = phytodf['ASP Algae'].groupby(phytodf['Location']).std()
sdsp = phytodf['ASP Algae'].groupby(phytodf['Location']).std()
spsp = phytodf['ASP Algae'].groupby(phytodf['Location']).std()
tpsp = 40
tasp = 1500
tdsp = 100
plt.figure(figsize=(15,5))
x_axis = np.arange(len(xlabels))
plt.bar(x_axis - 0.3, yall, label='All Phyto', width=0.2)
plt.bar(x_axis - 0.1, ypsp, yerr=spsp, label='PSP Algae', width=0.2) 
plt.bar(x_axis + 0.1, yasp, yerr=sasp, label='ASP Algae', width=0.2) 
plt.bar(x_axis + 0.3, ydsp, yerr=sdsp, label='DSP Algae', width=0.2)
plt.axhline(tpsp, linestyle='--', color='darkred')
plt.axhline(tasp, linestyle='--', color='darkred')
plt.xticks(x_axis, xlabels)
plt.style.use('tableau-colorblind10')
plt.annotate('Highly distributed data',
            xy=(2, 1), xycoords='data',
            xytext=(0.85, 0.63), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.98"))
plt.grid(True, alpha=0.3, )
plt.xlabel('Location', size=10)
plt.ylabel('Mean cells/l', size=10)
plt.title('Mean of Phytoplankton by Location')
plt.legend()
plt.show() 

# Plotting distribution of algal species by location
plt.figure(figsize=(15,5))
plt.bar(x_axis - 0.3, yall, label='All Phyto', width=0.2)
plt.bar(x_axis - 0.1, ypsp, label='PSP Algae', width=0.2) 
plt.bar(x_axis + 0.1, yasp, label='ASP Algae', width=0.2) 
plt.bar(x_axis + 0.3, ydsp, label='DSP Algae', width=0.2)
plt.axhline(tpsp, linestyle='--', color='darkred')
plt.axhline(tasp, linestyle='--', color='darkred')
plt.axhline(tdsp, linestyle='--', color='darkred')
plt.xticks(x_axis, xlabels)
plt.style.use('tableau-colorblind10')
plt.annotate('Belfast, Dundrum Bay, Larne\nLough Strangford with mean\ntoxicity above threshold level',
            xy=(2, 1), xycoords='data',
            xytext=(0.81, 0.6), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.98"))
plt.annotate('Toxic ASP\nThreshold',
            xy=(2, 1), xycoords='data',
            xytext=(0.94, 0.22), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.98"))
plt.annotate('Toxic PSP/DSP\nThreshold',
            xy=(2, 1), xycoords='data',
            xytext=(0.94, 0.03), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.98"))
plt.grid(True, alpha=0.3, )
plt.xlabel('Location', size=10)
plt.ylabel('Mean mg/kg', size=10)
plt.title('Mean of Phytoplankton by Location')
plt.legend()
plt.show() 

# Plot distribution as a percentage of total phytoplankton by location
# Creating the empty dictionaries
perc_phytoPSP = {}
perc_phytoASP = {}
perc_phytoDSP = {}

# loading standardised data into empty dictionaries
for location, df in dataframes.items():
    perc_phytoPSP[location] = (sumPSPphyto[location] / sumAllphyto[location] ) *100
    perc_phytoASP[location] = (sumASPphyto[location] / sumAllphyto[location] ) *100
    perc_phytoDSP[location] = (sumDSPphyto[location] / sumAllphyto[location] ) *100

# Create index mapping for each location
index = {'Belfast': 1, 'Carlingford': 2, 'Dundrum Bay': 3, 'Killough': 4, 'Larne Lough': 5, 'Lough Foyle': 6, 'Strangford': 7}

# Plot stacked bar plot
plt.figure(figsize=(15,5))
for location in dataframes.keys():
    if location != 'All':
        plt.bar(index[location], perc_phytoPSP[location], bottom=perc_phytoDSP[location], width=0.2, align='center', color='firebrick')
        plt.bar(index[location], perc_phytoDSP[location], bottom=perc_phytoASP[location], width=0.2, align='center', color='yellow')
        plt.bar(index[location], perc_phytoASP[location], width=0.2, align='center', color='cornflowerblue')
    else:
        None
plt.xticks([1, 2, 3, 4, 5, 6, 7],['Belfast', 'Carlingford', 'Dundrum Bay', 'Killough', 'Larne Lough', 'Lough Foyle', 'Strangford'])
plt.annotate('DSP Species are \nmost abundant',
            xy=(2, 1), xycoords='data',
            xytext=(0.89, 0.69), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="0.98"))
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Location', size=10)
plt.ylabel('Percentage of Species Found', size=10)
plt.title('Phytoplankton by Location')
plt.legend(['PSP', 'ASP', 'DSP'])
plt.show()
```
### **Timeseries Plot and Outlier Removal**
Plotting the sum of algal species concentration by location over time reveals these anomalies, for example the Larne Foyle high distribution is likely caused by the peak in spring 2021 at ~500000. The previous three-month average has been used to impute this Larne Foyle anomaly.

Imputation has been used to replace values above the 99th percentile with the 99th percentile to remove outliers. This allows for removal of extreme values whilst maintaining some stochasticity.
```Python
# Sort by Date_MY
phytodf = phytodf.sort_values(by='Date_MY')

# Create 100th percentile for use in Larne Foyle anomaly imputation
percentile_100 = np.percentile(phytodf['All Algae'], 100)
print(percentile_100)

# Impute Larne Foyle anomaly through use of previous month rolling 3 month average
phytodf['3_month_avg'] = phytodf['All Algae'].rolling(window=3).mean()
phytodf.loc[phytodf['All Algae'] == percentile_100, 'All Algae'] = phytodf['3_month_avg'].shift(1)

# Create 99th percentile for identification of outliers
percentile_99 = np.percentile(phytodf['All Algae'], 99)

# Plot algal concentrations over time by location grouped by algal type to see outliers
grouped = phytodf.groupby('Location')
plt.figure(figsize=(15,5))
for name, group in grouped:
    plt.plot(group['Date_MY'], group['All Algae'], label=name)
plt.axhline( percentile_99, linestyle='--', color='grey')
plt.annotate('99th Percentile',
             (phytodf['Date_MY'].iloc[1], percentile_99),
             textcoords="offset points", xytext=(0,10), ha='center',
             bbox=dict(boxstyle="round", fc="0.98")
            )
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Date', size=10)
plt.ylabel('Algal Concentration', size=10)
plt.title('Algal Concentration by Date and Location')
plt.show() 


# Impute outliers above 99th percentile to be equal to 99th percentile value
phytodf['All Algae'] = np.where(phytodf['All Algae'] > percentile_99, percentile_99, phytodf['All Algae'])

# Plot transformed data by location grouped by algal type
grouped = phytodf.groupby('Location')
plt.figure(figsize=(15,5))
for name, group in grouped:
    plt.plot(group['Date_MY'], group['All Algae'], label=name)
#plt.axhline( percentile_95, linestyle='--', color='grey')
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Date', size=10)
plt.ylabel('Algal Concentration', size=10)
plt.title('Algal Concentration by Date and Location - 99th Percentile Imputated')
plt.show()
```
### **Data Transformation for Timeseries Forecasting**
The result demonstrates similar seasonal behaviour for all locations with algal concentrations highest in summer months and lowest in winter. Therefore, location will be disregarding during the timeseries forecasting with the data grouped by mean of each month.  This removes any further anomalies and allows for prediction of algal concentrations around Ireland regardless of species or specific fishery.

Timeseries forecasting will be used to predict algal concentrations over the next 12 months.
```Python
# Plot Mean Algal Concentration by Location and Date
locations = phytodf['Location'].unique()
print(locations)
plt.figure(figsize=(15,5))
for location in locations:
    subset = phytodf[phytodf['Location'] == location]
    subsetmeans = subset.groupby('Date_MY')['All Algae'].mean()
    plt.plot(subset['Date_MY'].unique(), subsetmeans.values, label=f'{location}')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Date', size=10)
plt.ylabel('Mean Algal Concentration', size=10)
plt.title('Mean Algal Concentration by Date and Location')
plt.legend() 
plt.show() 

# Group phytodf by location by taking of all algae for all locations
dfmeanall = phytodf.groupby('Date_MY')['All Algae'].mean().reset_index()

# Plot Mean Algal Concentration by Date
plt.figure(figsize=(15,5))
plt.plot(phytodf['Date_MY'].unique(), dfmeanall['All Algae'].values)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Date', size=10)
plt.ylabel('Mean Algal Concentration', size=10)
plt.title('Mean Algal Concentration by Date')
plt.show()

# Rename dfmeanall column for ease of use in model
dfmeanall.rename(columns={ 'All Algae':'algaemean'}, inplace=True)

# Set index to date column
dfmeanall = dfmeanall.set_index('Date_MY')

# View data
print(dfmeanall.head())
```
## Timeseries Forecasting
A SARIMA timeseries forecast will be used as the predictive method with the hyperparameters (p,d,q)x(P,D,Q)s. A SARIMA forecast has been chosen over more complex machine learning algorithms such as XGBoost as the data is univariate, with a clear seasonal pattern and doesn’t require multi-variate or complex training 

### **SARIMA Stationarity Test**
First step in applying a successful SARIMA forecast is checking the data for stationarity which is represented by Integrated part of ARIMA defined by the d hyperparameter. This was achieved through use of a dickey fuller test, the outputs of which revealed a p-value > 0.05 threshold indicating the algal concentration over time is already stationary. A rolling mean and standard deviation were also plotted to help visualise this stationarity which also showed little variation over time,. The optimum d and D hyperparameter will likely be 0 as no differencing is required.
```Python
# ADF test to test stationarity
dftest = adfuller(dfmeanall.algaemean, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

# Plot rolling mean and standard deviation to view stationarity
rolling_mean = dfmeanall.rolling(window=12).mean()
rolling_std = dfmeanall.rolling(window=12).std()

plt.figure(figsize=(15,5))
plt.plot(dfmeanall, label="Original")
plt.plot(rolling_mean, label="Rolling Mean", color="red")
plt.plot(rolling_std, label="Rolling Std", color="black")
plt.legend()
plt.show()
```

### **Autocorreletation plots for p,d,q,P,D,Q,s determination**
Next, the autocorrelation (ACF) and partial-autocorrelation (PACF) where plotted to help determine the p,d,q,s values of the SARIMA. The ACF relates to the moving average p hyperparameter. The ACF plots reveals a significant spike at 1 and a potential seasonal component of 12-13 based on lags, indicating q-value may be 1, with seasonal s-value of 12-13. The PACF relates to the autoregressive p hyperparameter. The PACF plot reveals outliers at 1 with no significant lags indicating p-value may be 1.
```Python
# Autocorrelation plot
autocorrelation_plot(dfmeanall)
plt.title('Autocorrelation')
plt.show()

# ACF plot
plot_acf(dfmeanall)
plt.title('ACF')
plt.show()

# PACF plot
plot_pacf(dfmeanall)
plt.title('PACF')
plt.show()

# Decomposition plot
decomposition=seasonal_decompose(dfmeanall,model='additive',period=12)
decomposition.plot()
plt.show()
```
Manual review of ADF, ACF and PACF result in a probable SARIMA model of (1, 0, 0)x(0, 1, 1)12

### **Hyperparameter Optimisation**
To further determine the best values for the hyperparameters grid search has been used for hyper-parameter optimisation. This assesses all hyper-parameter combinations over a given range and returns the optimised combination based on three outputs; mean-squared error, mean-absolute error, Akaike Information Criterion (AIC). Mean-absolute error and mean-squared error measure the average difference and average of squares between predicted and actuals as an indication of prediction accuracy. AIC measures how effectively the model applied fits the actuals with low AIC being a better fit.
```Python
# Create class for testing each hyperparameter combinations against evaluators
class SARIMAgridsearch:
    def __init__(testing, data, p_values, d_values, q_values, P_values, D_values, Q_values, s_values): # initialise the class
        testing.data = data
        testing.p_values = p_values
        testing.d_values = d_values
        testing.q_values = q_values
        testing.P_values = P_values
        testing.D_values = D_values
        testing.Q_values = Q_values
        testing.s_values = s_values

# Fit the SARIMA model with the iniitalised variables
    def fit_model(testing, order, seasonal_order):
        sarima_model = SARIMAX(testing.data, order=order, seasonal_order=seasonal_order) # fit model
        results = sarima_model.fit(disp=False) # minimise output
        return results

# Define evaluation method
    def evaluate_model(testing, order, seasonal_order):
        results = testing.fit_model(order, seasonal_order)
        forecast = results.get_forecast(steps=12) # test next 12 months
        forecast_mean = forecast.predicted_mean
   #    mse = mean_squared_error(testing.data[-12:], forecast_mean)
    #   return mse
        mae = mean_absolute_error(testing.data[-12:], forecast_mean)
        return mae
   #    return results.aic ## AIC results

# Perform the grid search using the various hyperparameter combinations and evaluation method
    def grid_search(testing):
        best_score, best_pdq = float("inf"), None
        param_grid = ParameterGrid({
            'p': testing.p_values,
            'd': testing.d_values,
            'q': testing.q_values,
            'P': testing.P_values,
            'D': testing.D_values,
            'Q': testing.Q_values,
            's': testing.s_values
        })
        for params in param_grid:
            order = (params['p'], params['d'], params['q'])
            seasonal_order = (params['P'], params['D'], params['Q'], params['s'])
            try:
                optimum = testing.evaluate_model(order, seasonal_order)
                if optimum < best_score: # and optimum > 11: ## Needed for AIC, difference in Work and Home Jupyter kernels due to Python versions
                    best_score, best_pdq = optimum, (order, seasonal_order)
            except:
                continue
        print(f'Best SARIMA{best_pdq[0]}x{best_pdq[1]} with lowest AIC:{best_score}')
        return best_pdq

# Apply phyto data and hyperparameter ranges
data = pd.Series(dfmeanall['algaemean'])
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
P_values = [0, 1, 2]
D_values = [0, 1]
Q_values = [0, 1, 2]
s_values = [12, 13]

## test combination
# p_values = [0, 1]
# d_values = [0, 1]
# q_values = [0, 1]
# P_values = [0, 1]
# D_values = [0, 1]
# Q_values = [0, 1]
# s_values = [12]

# Return best hyperparameter comination
sarima_search = SARIMAgridsearch(data, p_values, d_values, q_values, P_values, D_values, Q_values, s_values)
best_order, best_seasonal_order = sarima_search.grid_search()
```
### The hyperparameter optimisation was performed 3 times with 3 different evaluation methods; AIC, MAE and MSE

### The outcome of the three models were:
    AIC: Best SARIMA(0, 1, 1)x(0, 1, 2, 12) with lowest AIC:911.0409793990797
    MAE: Best SARIMA(0, 0, 0)x(0, 1, 0, 12) with lowest MAE:1.0066022089934752e-13
    MSE: Best SARIMA(0, 0, 0)x(0, 1, 0, 12) with lowest MSE:2.6135487697242267e-26

MAE and MSE result in the same SARIMA model. The AIC and MAE/MSE optimised SARIMA models and the model assumed from manual ACF/PACF review were each plotted to forecast algal concentrations over the next 12 months with residuals and results.

### **Best AIC Model**
```Python
# Plot AIC model
model = SARIMAX(dfmeanall, order=(0, 1, 1), seasonal_order=(0, 1, 2, 12)) 
results = model.fit()

# Forecast for next 12 months
forecast_periods = 12 
forecast = results.get_forecast(steps=forecast_periods)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(dfmeanall, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.title("Forecasted Algal Concentration")
plt.xlabel("Date")
plt.ylabel("Algal Concentraion cells/L")
plt.legend()
plt.show()

# Print results
print(results.summary())

# Get residuals for testing of model effectiveness
residuals = results.resid
print(residuals)

# Plot residuals as ACF and PACF
plot_acf(residuals)
plot_pacf(residuals)
plt.show()

residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

# Calculate MAE and MSE for testing of model effectiveness
observed = dfmeanall[-forecast_periods:]
mae = mean_absolute_error(observed, forecast_mean)
mse = mean_squared_error(observed, forecast_mean)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
```

### **Best MAE Model**
```Python
# Plot MAE model 
model = SARIMAX(dfmeanall, order=(0, 0, 0), seasonal_order=(0, 1, 0, 12))
results = model.fit()

# Forecast for next 12 months
forecast_periods = 12 
forecast = results.get_forecast(steps=forecast_periods)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(dfmeanall, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.title("Forecasted Algal Concentration")
plt.xlabel("Date")
plt.ylabel("Algal Concentraion cells/L")
plt.legend()
plt.show()

# Print results
print(results.summary())

# Get residuals for testing of model effectiveness
residuals = results.resid
print(residuals)

# Plot residuals as ACF and PACF
plot_acf(residuals)
plot_pacf(residuals)
plt.show()

residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

# Calculate MAE and MSE for testing of model effectiveness
observed = dfmeanall[-forecast_periods:]
mae = mean_absolute_error(observed, forecast_mean)
mse = mean_squared_error(observed, forecast_mean)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
```

### **Best MSE Model**
```Python
# Plot MSE model
model = SARIMAX(dfmeanall, order=(0, 0, 0), seasonal_order=(0, 1, 0, 12))
results = model.fit()

# Forecast for next 12 months
forecast_periods = 12 
forecast = results.get_forecast(steps=forecast_periods)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(dfmeanall, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.title("Forecasted Algal Concentration")
plt.xlabel("Date")
plt.ylabel("Algal Concentraion cells/L")
plt.legend()
plt.show()

# Print results
print(results.summary())

# Get residuals for testing of model effectiveness
residuals = results.resid
print(residuals)

# Plot residuals as ACF and PACF
plot_acf(residuals)
plot_pacf(residuals)
plt.show()

residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

# Calculate MAE and MSE for testing of model effectiveness
observed = dfmeanall[-forecast_periods:]
mae = mean_absolute_error(observed, forecast_mean)
mse = mean_squared_error(observed, forecast_mean)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
```
### **Best Manual Model**
```Python
# Plot Manual model
model = SARIMAX(dfmeanall, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12))
results = model.fit()

# Forecast for next 12 months
forecast_periods = 12 
forecast = results.get_forecast(steps=forecast_periods)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(dfmeanall, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.title("Forecasted Algal Concentration")
plt.xlabel("Date")
plt.ylabel("Algal Concentraion cells/L")
plt.legend()
plt.show()

# Print results
print(results.summary())

# Get residuals for testing of model effectiveness
residuals = results.resid
print(residuals)

# Plot residuals as ACF and PACF
plot_acf(residuals)
plot_pacf(residuals)
plt.show()

residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

# Calculate MAE and MSE for testing of model effectiveness
observed = dfmeanall[-forecast_periods:]
mae = mean_absolute_error(observed, forecast_mean)
mse = mean_squared_error(observed, forecast_mean)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
```
### **Overview**
Due to the small size of the data, test and train datasets have not been created. The SARIMA model has been trained on the entire dataset and is at high risk of overfitting. Of the three attributes grid search assess against, only AIC penalises for overfitting, therefore this model should be recommended moving forward. The MSE and MAE do not penalise overfitting, consequently it is likely the model has been trained to replicate the data and not learnt the distribution 
