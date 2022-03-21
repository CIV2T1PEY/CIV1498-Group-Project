# ----------------Import packages
import os
import json
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
import matplotlib.pylab as plt

# ----------------Panda Dataframe Set Up
SBA_Loan = pd.read_csv('SBAnational.csv', low_memory=False)
SBA_Loan = SBA_Loan.sort_values(by='ApprovalFY', ascending=True)
SBA_Loan['ApprovalDate'] = pd.to_datetime(SBA_Loan['ApprovalDate'])
SBA_Loan_TX = SBA_Loan[SBA_Loan['State'] == 'TX']


US_President = pd.read_csv('US_President.csv')

# print(SBA_Loan.info())  #check column names
# print(US_President.info())  #check column names

# print(SBA_Loan['ApprovalFY'].values.min())  #check the earliest date, output = 1962
# print(SBA_Loan['ApprovalFY'].values.max())  #check the latest date, output = 2014

US_President['start date'] = pd.to_datetime(US_President['From'])

US_President['end date'] = pd.to_datetime(US_President['To'],errors='coerce')  # the last end date is current, which is a string, replace it with NAT
US_President['end date'] = US_President['end date'].fillna(pd.to_datetime('2021-01-20'))  # Replace NAT with the googled last day of Trump's presidency term

#----------------Need to drop the dollar under some columns in SBA_Loan and convert the type to float
def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '')
    return x

SBA_Loan['DisbursementGross'] = SBA_Loan['DisbursementGross'].apply(clean_currency).astype('float')
SBA_Loan['ChgOffPrinGr'] = SBA_Loan['ChgOffPrinGr'].apply(clean_currency).astype('float')
SBA_Loan['GrAppv'] = SBA_Loan['GrAppv'].apply(clean_currency).astype('float')
SBA_Loan['SBA_Appv'] = SBA_Loan['SBA_Appv'].apply(clean_currency).astype('float')
print(SBA_Loan['DisbursementGross'])

# --------------Create a function for merging based timestamp between two dates, run time is long, optimization could occur
def merge_on_date(df_A, df_B, df_A_column, df_B_column):
    for index, row in df_B.iterrows():
        start_date = row['start date']
        end_date = row['end date']
        for index1, row1 in df_A.iterrows():
            if row1['ApprovalDate'] < end_date and row1['ApprovalDate'] >= start_date:
                df_A.at[index1,df_A_column] = row[df_B_column]

    return df_A

#---------------Testing the function above
#SBA_Loan_TX = merge_on_date(SBA_Loan_TX,US_President,'President','President')
SBA_Loan_TX = merge_on_date(SBA_Loan_TX,US_President,'President Party','Party')
print(SBA_Loan_TX.head())
#print(SBA_Loan_TX.info())

#------------------Visualization
SBA_Loan_TX_grouped = SBA_Loan_TX.groupby('President Party')['DisbursementGross'].sum()
print(SBA_Loan_TX_grouped)


