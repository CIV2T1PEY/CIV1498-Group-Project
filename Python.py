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

SBA_Loan['ApprovalDate'] = pd.to_datetime(SBA_Loan['ApprovalDate'], dayfirst=True, format='%d-%b-%y')
#SBA_Loan = SBA_Loan.sort_values(by='ApprovalFY', ascending=False)


US_President = pd.read_csv('US_President.csv')

#print(SBA_Loan['ApprovalFY'].values.min())  #check the earliest year, output = 1962
#print(SBA_Loan['ApprovalFY'].values.max())  #check the latest year, output = 2014
print(SBA_Loan['ApprovalDate'].values.min())  #check the earliest date, output = 1969-01-31, doesnt match with the approval year data
print(SBA_Loan['ApprovalDate'].values.max())  #check the latest date, output = 2068-12-03, makes no sense, need to check to_datetime function above, it appears to_datetime function assumes year 62 to be 2062, but it is 1962



US_President['start date'] = pd.to_datetime(US_President['From'])

US_President['end date'] = pd.to_datetime(US_President['To'],errors='coerce')  # the last end date is current, which is a string, replace it with NAT
US_President['end date'] = US_President['end date'].fillna(pd.to_datetime('2021-01-20'))  # Replace NAT with the googled last day of Trump's presidency term

#----------------Overview of the Dataframe
print(SBA_Loan.info())
print(SBA_Loan['NAICS'].value_counts()[0]/SBA_Loan.shape[0])   #about 22.5% of data do not have industry info, they can be removed for some questions
print(SBA_Loan['Term'].value_counts()[0]/SBA_Loan.shape[0])    #about 0.1% of data have 0 term, which makes no sense, since the term is in months, it is likely that the loan duration is less than 1 month (e.g., 1 week), in this case 0 should be round up to 1


#Data Cleaning
#----------------Drop the dollar under some columns in SBA_Loan and convert the type to float
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
SBA_Loan['BalanceGross'] = SBA_Loan['BalanceGross'].apply(clean_currency).astype('float')
#print(SBA_Loan['DisbursementGross']) #test

#----------------round term up from 0 to 1
SBA_Loan['Term'] = SBA_Loan['Term'].replace(0, 1)

#----------------fix ApprovalFY column to remove alphabet characters
SBA_Loan['ApprovalFY'] = SBA_Loan['ApprovalFY'].apply(lambda x: ''.join(filter(str.isdigit,x)))

#----------------fix the year in ApprovalDate
def fix_year(series_A, series_B):
    empty_list = []
    for index, value in enumerate(series_A):
        k = value.replace(year = int(series_B[index]))
        empty_list.append(k)
    series_A = empty_list
    return series_A

SBA_Loan['ApprovalDate'] = fix_year(SBA_Loan['ApprovalDate'],SBA_Loan['ApprovalFY'])

print(SBA_Loan['ApprovalFY'].values.min())  #check the earliest year, output = 1962
print(SBA_Loan['ApprovalFY'].values.max())  #check the latest year, output = 2014
print(SBA_Loan['ApprovalDate'].values.min())  #check the earliest date, output = 1962-12-07, code worked!
print(SBA_Loan['ApprovalDate'].values.max())  #check the latest date, output = 2014-12-31, code worked!


#----------------convert Industry codes to text
NAICS_DF = pd.read_csv('General Industry Codes.csv')
NAICS_DF = NAICS_DF.set_index('Code')
print(NAICS_DF.info())
print(list(NAICS_DF.index.values))

def industry_identifier(code):
    industry_name = 'Undefined'
    first_two_digits = int(str(code)[:2])
    if first_two_digits in NAICS_DF.index.values:
        if code != 0:
           #industry_name = NAICS_DF[NAICS_DF['Code'] == first_two_digits]['IndustryName'].values
           industry_name = NAICS_DF.loc[first_two_digits, 'IndustryName']

    return industry_name

SBA_Loan['Industry'] = SBA_Loan['NAICS'].apply(industry_identifier)

SBA_Loan.to_csv('SBA_Loan_test.csv', index=False)
#Feature Engineering
# --------------Create a function for merging based timestamp between two dates, run time is long, optimization could occur
def merge_on_date(df_A, df_B, df_A_column, df_B_column):
    for index, row in df_B.iterrows():
        start_date = row['start date']
        end_date = row['end date']
        for index1, row1 in df_A.iterrows():
            if row1['ApprovalDate'] < end_date and row1['ApprovalDate'] >= start_date:
                df_A.at[index1, df_A_column] = row[df_B_column]

    return df_A

#---------------Testing the function above
#SBA_Loan_TX = SBA_Loan[SBA_Loan['State'] == 'TX']
SBA_Loan_FL = SBA_Loan[SBA_Loan['State'] == 'FL']       #focus on Florida since the entire dataset is too big, Florida has historically being a swing state
#SBA_Loan_TX = merge_on_date(SBA_Loan_TX, US_President, 'President Party','Party')
SBA_Loan_FL = merge_on_date(SBA_Loan_FL, US_President, 'President Party','Party')
#SBA_Loan_merged = merge_on_date(SBA_Loan, US_President, 'President Party','Party')
#SBA_Loan_TX = SBA_Loan[SBA_Loan['State'] == 'TX']
#print(SBA_Loan_TX.head())
#print(SBA_Loan_TX.info())
#SBA_Loan_TX.to_csv('SBA_Loan_TX.csv', index=False)
SBA_Loan_FL.to_csv('SBA_Loan_FL.csv', index=False)

#------------------Visualization
#SBA_Loan_TX_grouped = SBA_Loan_TX.groupby('President Party')['DisbursementGross'].sum()       #output the total amount of loan approved with each of the two parties, has to be normalized to amount per month/year
#print(SBA_Loan_TX_grouped)


