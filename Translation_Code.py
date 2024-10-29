#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:35:12 2023

@author: katherinemartin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:35:53 2023

@author: katherinemartin
"""
import pandas as pd
import numpy as np
import json
import datetime
import os
os.chdir('/Users/katherinemartin/Documents/SOI Data/Python/SOI_Qualtrics_SOP')
import QualAPI as qa
import requests
import re
from collections import Counter
from googletrans import Translator
#pip freeze | grep googletrans
#!!pip3 install googletrans==4.0.0rc1

import time
import emoji
import warnings
import httpx
import nltk
# nltk.download('all') # nltk.download('vader_lexicon')
from textblob import TextBlob
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
pd.options.mode.chained_assignment = None 
###############################################################################
# Inputs for wrapper function 
# Opening JSON file

f = open('/Users/katherinemartin/Documents/SOI Data/Python/Qualtrics SOP/qualtrics_credentials.txt')
# read the json
creds = json.load(f)

with open('/Users/katherinemartin/Documents/SOI Data/Python/Qualtrics SOP/qualtrics_credentials.txt', 'r') as f:
    creds = json.load(f)
    print(creds)
    
# Extract the clientId, client Secret, and dataCenter
clientId = creds.get('ID')
clientSecret = creds.get('Secret')
dataCenter = creds.get('DataCenter')

# Extract the Survey ID based on the name of the Survey
#SurveyName = 'Community Health Worker (CHW) Training: Pre-Training Survey'
SurveyName = 'World Summer Games Berlin 2023 Athlete Satisfaction Survey'
# SurveyName = 'Current Y2 HIG Health Evaluation Report 2022-23'

###############################################################################
# create the base url 
base_url = 'https://{0}.qualtrics.com'.format(dataCenter)

# set up the data for the token
grant_type = 'client_credentials'
#scope = 'read:surveys read:survey_responses manage:all manage:users read:users write:users'
scope = 'read:surveys read:survey_responses'
data = qa.DefDataT(grant_type = grant_type, scope = scope)

# Get the token
Btkn = qa.getToken(base_url, clientId, clientSecret, data)
# Extract the Bearer access token
tkn = Btkn.get("access_token")

# Pull the list of surveys
SurveyDF = qa.getSurveys(base_url, tkn)

# Pull the indices associated with the Survey Name
Inds = SurveyDF.loc[SurveyDF['name'] == SurveyName].index[:]

if len(Inds) > 1:
    print('Multiple Surveys Have the Same Name!!!')
elif len(Inds) == 0:
    print('Cannot Find the Survey. Please Check that the Survey Name is Correct.')
elif len(Inds) == 1:
    Indx = Inds[0]    

# Extract the Survey Id 
SurveyId = SurveyDF.loc[Indx, 'id']

# Get the Response Export
RX = qa.getSurveyRX(base_url, tkn, SurveyId)
# Get the exportProgressId
EPid = RX.get('result').get('progressId')

# wait until the status is 100% complete
# set fid to blank
fid = ""
while len(fid) == 0:    
    # Check the progress 
    RXp = qa.getSurveyRXp(base_url, tkn, SurveyId, EPid)

    # # Check if the percent complete is 100
    if RXp.get('result').get('percentComplete') == 100:
        # Then check to see if the pull is complete
        if RXp.get('result').get('status') == 'complete':
            # Get the file Id
            fid = RXp.get('result').get('fileId')
        else:
            print("Status: " + RXp.get('result').get('status'))

# Get the survey responses
Survey = qa.getSurvey(base_url, tkn, SurveyId, fid)
# Extract and arrange data
# Convert to JSON
SurveyJ = Survey.json()
# Get the list of responses
Responses = SurveyJ.get('responses')
if len(Responses) > 0:
    Results = qa.OrganizeResponses(Responses)
else:
    print('There is no data')

print(Responses)
###############################################################################
# do not keep survey preview  responses
KeepMask = np.array(Results['distributionChannel'] != 'Survey Preview')
Results = Results.loc[KeepMask, :]
# Reset the index
Results = Results.reset_index(drop = True)
# Results has user language column

# Get the survey questions
SurveyQs = qa.getSurveyQs(base_url, tkn, SurveyId)

# Map the Survey Qs so that they can match with column headers of Results
SurveyQsR = SurveyQs.get('result')

QDic = SurveyQsR.get('questions')

# Get the Column Data Types
QDF, QVals = qa.ExtractColumnDataTypes(QDic, Results, base_url, tkn, SurveyId)

# Get the Different Data Types for reporting purposes
QDF = qa.DicDataType(QDF, QVals)

###############################################################################
# I am reordering how you were doing your calculations and cleaning some 
# processes to build in greater resiliency

# Using the QDF you can extract only those columns with questions
df = Results.loc[:, QDF['QID'].tolist()]

NaNMask = df.isna()
KeepMask = np.array(NaNMask.sum(axis = 1) < len(df.columns))

# Determine the rows that have no data... (i.e., can 

# Now adjust Results and the df
Results = Results.loc[KeepMask, :]
df = df.loc[KeepMask, :]

# Reset the index
Results = Results.reset_index(drop = True)
df = df.reset_index(drop = True)


###############################################################################
# Determine the percent of missing data by question
NaNMask = df.isna()
MissingDataN = NaNMask.sum(axis = 0).tolist() 
PresentDataN = len(df) - np.array(MissingDataN)
# Get the fraction of respondents by question
MissingDataFrc = np.array(MissingDataN) / len(df)
PresentDataFrc = 1 - MissingDataFrc

# Create a data frame
tdic = {'QID': df.columns,
        "MissingDataN": MissingDataN,
        "PresentDataN": PresentDataN,
        "MissingDataFrc": MissingDataFrc,
        "PresentDataFrc": PresentDataFrc}

MissingDataDF = pd.DataFrame(tdic)

FreeTextCols = QDF.QID[np.array(QDF.DataType == 'FreeText')]
FreeTextCols = FreeTextCols[~FreeTextCols.str.contains('TEXT_')]

# Create text_df and use for translation or clean and process df to use for 
# translation:
text_df = df[FreeTextCols]
# Assuming QDF is your DataFrame with columns 'QID' and 'DataType'
def process_text_columns(df, free_text_cols):
    """ Process specified text columns in a dataframe.

    Parameters:
    -----------
    df : DataFrame
        The dataframe to process.
    free_text_cols : Series
        Series containing the names of the columns to process.
    """
    # Identify columns in df that are also in free_text_cols
    cols_to_process = df.columns.intersection(free_text_cols)

    for col in cols_to_process:
        # Convert to string
        df[col] = df[col].astype(str)
        # Replace 'N/A' and 'nan' with NaN
        df[col] = df[col].replace(r'N/A', np.nan, regex=True).replace(r'nan', np.nan, regex=True)

        # Replace new lines with space and whitespace strings with NaN
        df[col] = df[col].replace(r'\n', ' ', regex=True).replace(r'^\s*$', np.nan, regex=True)

        # Convert to lowercase and remove commas and special characters
        df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()) if pd.notnull(x) else x)
        
    if df.equals(text_df):
        # Drop columns that are all null only if df is not equal to text_df
        df.dropna(axis=1, how='all', inplace=True)

    return df

# Can clean either full df or just text_df depending on what function
# will be run below
text_df = process_text_columns(text_df, FreeTextCols)
df = process_text_columns(df, FreeTextCols)


# Every data set will have some unique cleaning that must be done. That can
# go here:
text_df.replace('Deutschland', 'Germany', inplace=True)



# Honestly dont know if we need or want this. Striping capital letters and 
# special characters seem to be more effective and when it comes to athletes
# emojis are important for helping them communicate

# def remove_emojis(df):
#     """ Function to remove emojis from a dataframe
#     Parameters:
#         -----------
#         df (dataframe): dataframe including any columns
#     """
#     # get string columns
#     non_numeric_columns = df.convert_dtypes().select_dtypes("string")
#     non_numeric_columns = list(non_numeric_columns.columns)
    
#     # remove emojis from each string column
#     for col in non_numeric_columns:
#         df[col] = df[col].apply(
#             lambda s: emoji.replace_emoji(s, '')
#             if pd.notnull(s) and s != "" and s != "\n" else s)
#     return df

# text_df = remove_emojis(text_df)
# df = remove_emojis(df)

######################## TRANSLATION ########################
# This function translated the text_df and creates a new translated columnn and
# an error column for every column in text_df. It does not map back to df
# and is good for testing/comparing the resutls.
def translate_seperate_text_df(df, col, translator):
    """ Helper function to translate a column in a dataframe.
    Parameters:
        -----------
        df (dataframe): dataframe including column to be translated
        col (str): specific column to be translated
    """
    translated_col_name = col + '_translated'
    error_col_name = col + '_error'
    df[translated_col_name] = pd.NA  # Initialize new column with NA
    df[error_col_name] = pd.NA  # Initialize error column

    for index, row in df.iterrows():
        if pd.notnull(row[col]) and row[col].strip(): # and row[col] != "" and row[col] != "\n":
            try:
                translated_text = translator.translate(row[col], src='auto', dest='en').text
                df.loc[index, translated_col_name] = translated_text
                time.sleep(0.5)
            except Exception as e:
                print("ERROR WITH:", row[col])
                df.loc[index, translated_col_name] = row[col]
                df.loc[index, error_col_name] = 'error'
        else:
            df.loc[index, translated_col_name] = row[col]
    
    # Reorder columns so that translated column is right after the original column
    col_index = df.columns.get_loc(col)
    reordered_cols = df.columns.to_list()[:col_index + 1] + [translated_col_name, error_col_name] + df.columns.to_list()[col_index + 1:-2]
    df = df[reordered_cols]

    return df

timeout = httpx.Timeout(120) 
translator = Translator(timeout=timeout)
translator.raise_Exception = True

translated_data_full = text_df.copy()
for col in FreeTextCols:
    print(col, ": there are", translated_data_full[col].count(), "non null entries to translate.")
    translated_data_full = translate_seperate_text_df(translated_data_full, col, translator)
    print("Finished", col)
    time.sleep(5)
    

# This function code takes all of the columns that match FreeTextCols from df
# translates them and then replaces them in df. There are no comparison columns
def translate_replace_full_df(df, col, translator):
    """ Helper function to translate a column in a dataframe in-place.
    Parameters:
        -----------
        df (dataframe): dataframe including column to be translated
        col (str): specific column to be translated
    """
    for index, row in df.iterrows():
        if pd.notnull(row[col]) and row[col].strip():
            try:
                translated_text = translator.translate(row[col], src='auto', dest='en').text
                df.at[index, col] = translated_text
                time.sleep(0.5)
            except Exception as e:
                print("ERROR WITH:", row[col])
                # Optionally, handle error case here, e.g., leave the text as is or mark it
                # df.at[index, col] = 'Translation Error'  # or leave it as is
    return df

timeout = httpx.Timeout(120) 
translator = Translator(timeout=timeout)
translator.raise_Exception = True

for col in FreeTextCols:
    print(col, ": there are", df[col].count(), "non null entries to translate.")
    df = translate_replace_full_df(df, col, translator)
    print("Finished", col)
    time.sleep(5)
 
# Code for the output csv or excel:
# df.to_csv('full_translation_athlete_survey.csv', index = False)
# 