#Import packages
import os
import json
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
import matplotlib.pylab as plt

#Panda Dataframe Set Up
#SBA_Loan = pd.read_csv('SBAnational.csv',low_memory = False)
US_President = pd.read_csv('US_President.csv')

#print(SBA_Loan.info())  #check column names
#print(US_President.info())  #check column names

#print(SBA_Loan['ApprovalFY'].values.min())  #check the earliest date, output = 1962
#print(SBA_Loan['ApprovalFY'].values.max())  #check the latest date, output = 2014

US_President['start date'] = pd.to_datetime(US_President['From'])

US_President['end date'] = pd.to_datetime(US_President['To'],errors = 'coerce') #the last end date is current, which is a string, replace it with NAT
US_President['end date'] = US_President['end date'].fillna(pd.to_datetime('2021-01-20')) #Replace NAT with the googled last day of Trump's presidency term


print(US_President['end date'])
