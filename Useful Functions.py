def industry_identifier(code):
    industry_name = 'Undefined'
    first_two_digits = int(str(code)[:2])
    if first_two_digits in NAICS_DF.index.values:
        if code != 0:
           #industry_name = NAICS_DF[NAICS_DF['Code'] == first_two_digits]['IndustryName'].values
           industry_name = NAICS_DF.loc[first_two_digits, 'IndustryName']

    return industry_name

def merge_on_date(df_A, df_B, df_A_column, df_B_column):
    for index, row in df_B.iterrows():
        start_date = row['start date']
        end_date = row['end date']
        for index1, row1 in df_A.iterrows():
            if row1['ApprovalDate'] < end_date and row1['ApprovalDate'] >= start_date:
                df_A.at[index1, df_A_column] = row[df_B_column]

    return df_A

def fix_year(series_A, series_B):
    empty_list = []
    for index, value in enumerate(series_A):
        k = value.replace(year = int(series_B[index]))
        empty_list.append(k)
    series_A = empty_list
    return series_A

def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '')
    return x