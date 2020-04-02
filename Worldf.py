""" 
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. The 
coronavirus has spread rapidly throughtout the world and while lot of reserch is being done on its origin, 
rate of spread and anatomy of the virus, very little is understood about why country specific numbers are 
turning out to be different. While the impact of countries like South Korea, Japan has been lower on cases, 
deaths in few countries like Germany have been lower even though cases are higher. There is grossly uneven 
metrics in terms of rate of infection, death rate, growth rate etc. This model attempts to discover if there 
is any connection between non intuitive features of a country and the number of cases and the number of deaths.
In Covid Analysis 1, i will leave the readers with what features are dominantly impacting the cases and deaths. 
This analysis will be focussing on what and not why or how much. I will try to cover why or how much
in the next article. 

First we will import all the standard and specific libraries which are required for doing this analysis. The 
library list includes standard imports like Pandas and Numpy, Stats Import like scipy and sklearn, Plots import 
like Matplotlib, seaborn and plotly. The model imports are done in the specific sections of the code"""

#Standard Import
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from numpy.random import randn
 
#Stats Import
from scipy import stats
import sklearn
from sklearn.datasets import load_boston

#Plots Import
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import plotly as pl
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()

#Other Imports
import re
import datetime

""" I follow a standard approach to all ML/AI models which includes 8 steps as follows:-
1) Ask Question
2) Get Data
3) Clean Data
4) Perform EDA
5) Preprocess Data
6) Split Data
7) Apply ML/AI
8) Inference
"""

#---------------------------------------------------------------------------------------------------
#1 Ask Question - We are doing risk analytics by predicting the probability of default for a tenant.
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
#2 Get Data -  The data is available in the Covid19Cases.xlsx  but more derived columns need to be added
#---------------------------------------------------------------------------------------------------

"""First we will get the latest coronavirus cases from the offcial European Centre for Disease Prevention 
and Control website. https://www.ecdc.europa.eu/. We are using the read csv method which is available in 
Pandas library"""

url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
Casesdf = pd.read_csv(url, encoding='ISO-8859-1')
Casesdf = Casesdf.rename(columns={'countriesAndTerritories': 'Country'})
Casesdf = Casesdf.drop('popData2018', axis=1)
Casesdf['dateRep'] = pd.to_datetime(Casesdf['dateRep'])

def replaceus(country):
      str = re.sub(r'[\W_]',' ', country)
      return(str)

Casesdf['CountryNew'] = Casesdf['Country'].apply(replaceus)
Casesdf = Casesdf.drop('Country', axis=1)
Casesdf = Casesdf.rename(columns={'CountryNew':'Country'})

"""We also take the UN data which provides statistics on every country wrt to their growth, polpulation, 
demographics etc
"""
url = "https://github.com/vinayakbhakta9/Coronavirus/blob/master/Covid19CasesNew.xlsx?raw=true"
Worlddf = pd.read_excel(url,sheet_name='World')
Worlddf.info()

"""We also try to get the meat consumption pattern to see if countries food habits have anything to do 
with the spread of the virus in those countries"""

url = "https://github.com/vinayakbhakta9/Coronavirus/blob/master/WorldPoultry.xlsx?raw=true"
Meetdf = pd.read_excel(url,sheet_name='Sheet1')
Meetdf.info()

#Calculating Age of the virus in the respective country and rank orders then picks rank 1
CasedfFil = Casesdf[Casesdf['cases']>0]
Cdfgr1 = CasedfFil.groupby('Country')
CasedfFil['Rn'] = Cdfgr1['dateRep'].rank(method='min')
CasedfFil = CasedfFil[CasedfFil['Rn']==1]

#age calculation. First gets todays date and then substract with first date

def today_date():
    date=datetime.datetime.now().date()
    date=pd.to_datetime(date)
    return date

CasedfFil['Today'] = today_date()
CasedfFil['Diff'] = CasedfFil['Today'] - CasedfFil['dateRep']

#joining with the world df
CasedfFil = CasedfFil[['Country', 'Diff']]
Worlddf = Worlddf.join(CasedfFil.set_index('Country'), on='Country')
Worlddf['Age'] = pd.to_numeric(Worlddf['Diff'].dt.days, downcast='integer')
Worlddf = Worlddf.drop('Diff', axis=1)

#Adding Count of Cases and Count of deaths in Worlf df
Casedfcd = Casesdf.groupby(['Country']).sum()
Casedfcd = Casedfcd.drop(['day','month','year'], axis=1)
Worlddf = Worlddf.join(Casedfcd, on='Country')

#joining with the meet df
Worlddf = Worlddf.join(Meetdf.set_index('Entity'), on='Country')

"""
#avg of daily increase in the number of cases
#first applying a lag function
Casesdf['DateYear'] = pd.to_datetime(Casesdf['dateRep']).dt.to_period('m')
Casesper = Casesdf.groupby(['Country', 'DateYear']).sum()
Casesper = Casesper.drop(['day','month','year'], axis=1)
Casesper['Data_lag_Cases'] = Casesper.groupby(['Country'])['cases'].shift(1)
Casesper = Casesper.fillna(value=0)
Casesper['Inc_Cases'] = Casesper['cases'] - Casesper['Data_lag_Cases']

#avg of daily increase in the number of cases
#first applying a lag function
Casesper['Data_lag_Deaths'] = Casesper.groupby(['Country'])['deaths'].shift(1)
Casesper = Casesper.fillna(value=0)
Casesper['Inc_Deaths'] = Casesper['deaths'] - Casesper['Data_lag_Deaths']

#Calculating Case Dated
Casesperpivot = Casesper.pivot_table('Inc_Cases','Country', 'DateYear')
Casesperpivot.columns = ['Case_201912', 'Case_202001','Case_202002','Case_202003']
Casesperpivot = Casesperpivot.fillna(value=0)           

#Adding Case Dated in Worlf df
Worlddf = Worlddf.join(Casesperpivot, on='Country')

#Calculating Death Dated
Casesperpivot = Casesper.pivot_table('Inc_Deaths','Country', 'DateYear')
Casesperpivot.columns = ['Death_201912', 'Death_202001','Death_202002','Death_202003']
Casesperpivot = Casesperpivot.fillna(value=0)           

#Adding Case Dated in Worlf df
Worlddf = Worlddf.join(Casesperpivot, on='Country')
x_transformed = Worlddf.drop(['Country','Region','Cases','Deaths'], axis=1)"""

#---------------------------------------------------------------------------------------------------
#3 Clean Data - Data is already clean enough
#Lease Delinquent is a categorical variable which we can convert to numberical by using Label Encoder
# https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
#---------------------------------------------------------------------------------------------------

Worlddf.replace([np.inf, -np.inf], np.nan, inplace=True)
Worlddf.fillna(0, inplace=True)

#---------------------------------------------------------------------------------------------------
#4 EDA of data to understand the trend, distribution etc 
#---------------------------------------------------------------------------------------------------

#Finding feature importance on number of cases with the 53 features

from sklearn.ensemble import RandomForestRegressor
dfn=Worlddf.drop(['Country','Region','cases','deaths'], axis=1)
lsd=Worlddf['cases']
lsd=pd.get_dummies(lsd)
model = RandomForestRegressor(random_state=1, max_depth=10)
dfn=pd.get_dummies(dfn)
model.fit(dfn,lsd)

features = dfn.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-18:]  # top 10 features
plt.title('Feature Importances by Cases')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

compcasedf = pd.DataFrame( {"Column": dfn.columns, "importance": importances}).sort_values('importance', ascending=False)

#Finding feature importance on number of deaths with the 53 features

from sklearn.ensemble import RandomForestRegressor
dfn=Worlddf.drop(['Country','Region','cases','deaths'], axis=1)
lsd=Worlddf['deaths']
lsd=pd.get_dummies(lsd)
model = RandomForestRegressor(random_state=1, max_depth=10)
dfn=pd.get_dummies(dfn)
model.fit(dfn,lsd)

features = dfn.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-18:]  # top 10 features
plt.title('Feature Importances by Deaths')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

compdf = pd.DataFrame( {"Column": dfn.columns, "importance": importances}).sort_values('importance', ascending=False)

#---------------------------------------------------------------------------------------------------
#8 Inference
# In Medium.Com
#---------------------------------------------------------------------------------------------------
