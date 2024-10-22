#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:35:53 2023

@author: kieranmartin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches
from docx.shared import Pt
import json
import datetime
import os
#os.chdir('/Users/kieranmartin/Documents/SOI Data/Python/Qualtrics_API_Program')
os.chdir('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics_API_Program')

import QualAPI as qa
import requests
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
from collections import Counter
###############################################################################
# Inputs for wrapper function 
# Opening JSON file

#f = open('/Users/kieranmartin/Documents/SOI Data/Python/Qualtrics_API_Program/qualtrics_credentials.txt')

f = open('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics_API_Program\\copa_qualtrics_credentials.txt')
# read the json
creds = json.load(f)

with open('/Users/kieranmartin/Documents/SOI Data/Python/Qualtrics_API_Program/qualtrics_credentials.txt', 'r') as f:
    creds = json.load(f)
    print(creds)
    
# Extract the clientId, client Secret, and dataCenter
client_id = creds.get('ID')
client_secret = creds.get('Secret')
data_center = creds.get('DataCenter')

# Extract the Survey ID based on the name of the Survey
#SurveyName = 'Community Health Worker (CHW) Training: Pre-Training Survey'
SurveyName = "Spring 2024 COPA People's Academy: Post-Participation Survey"
# SurveyName = 'Current Y2 HIG Health Evaluation Report 2022-23'


###############################################################################

# create the base url 
base_url = 'https://{0}.qualtrics.com'.format(data_center)

# set up the data for the token
grant_type = 'client_credentials'
#scope = 'read:surveys read:survey_responses manage:all manage:users read:users write:users'
scope = 'read:surveys read:survey_responses'
data = qa.return_kwargs_as_dict(grant_type = grant_type, scope = scope)

# Get the token
Btkn = qa.get_token(base_url, client_id, client_secret, data)
# Extract the Bearer access token
tkn = Btkn.get("access_token")

# Pull the list of surveys
SurveyDF = qa.get_survey_list(base_url, tkn)

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
RX = qa.export_survey_responses(base_url, tkn, SurveyId)
# Get the exportProgressId
EPid = RX.get('result').get('progressId')

# wait until the status is 100% complete
# set fid to blank
fid = ""
while len(fid) == 0:    
    # Check the progress 
    RXp = qa.get_response_export_progress(base_url, tkn, SurveyId, EPid)

    # # Check if the percent complete is 100
    if RXp.get('result').get('percentComplete') == 100:
        # Then check to see if the pull is complete
        if RXp.get('result').get('status') == 'complete':
            # Get the file Id
            fid = RXp.get('result').get('fileId')
        else:
            print("Status: " + RXp.get('result').get('status'))

# Get the survey responses
Survey = qa.get_survey_responses(base_url, tkn, SurveyId, fid)
# Extract and arrange data
# Convert to JSON
SurveyJ = Survey.json()
# Get the list of responses
Responses = SurveyJ.get('responses')

if len(Responses) > 0:
    Results = qa.organize_responses(Responses)
else:
    print('There is no data')

print(Responses)
###############################################################################
# do not keep survey preview  responses
KeepMask = np.array(Results['distributionChannel'] != 'preview')
Results = Results.loc[KeepMask, :]
# Reset the index
Results = Results.reset_index(drop = True)
###############################################################################
# Results = Results[Results['progress'] >= 100]
# Results = Results[Results['QID3'] == 2]





# Get the survey questions
SurveyQs = qa.get_survey_questions(base_url, tkn, SurveyId)

# Map the Survey Qs so that they can match with column headers of Results
SurveyQsR = SurveyQs.get('result')

QDic = SurveyQsR.get('questions')

# Get the Column Data Types
QDF, QVals = qa.extract_column_data_types(QDic, Results, base_url, tkn, SurveyId)

# Get the Different Data Types for reporting purposes
QDF = qa.create_data_type_dictionary(QDF, QVals)

###############################################################################
# Using the QDF you can extract only those columns with questions
df = Results.loc[:, QDF['QID'].tolist()]

# This code will find all the nans. Can be used to clean and determine nan
# Frequencies
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
# Grab the missing data for the text
Mask = np.array(np.isin(MissingDataDF['QID'], list(FreeTextCols)))
TextMissDataDF = MissingDataDF.loc[Mask, :]


'QID138'
# Inquire about the date range of the data to pull and delete any data
# that falls outside the date range
# Have start and end date in the csv file name. Example: data_startdate_enddate.csv
# Change date column 'RecordedDate' to date format
# def subset_by_date_range(df, start_date, end_date):
#     # Ensure the 'recordedDate' column is in datetime format
#     df['recordedDate'] = pd.to_datetime(df['recordedDate'])

#     # Convert the start and end dates to datetime
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)

#     # Subset the DataFrame based on the date range
#     subset_df = df[(df['recordedDate'] >= start_date) & (df['recordedDate'] <= end_date)]

#     return subset_df


# df = subset_by_date_range(df, '2023-01-01', '2023-03-31')


# For each multiple choice question create a table with the frequency responses
# and fractions
MCcols = list(QDF.QID[np.array(QDF["DataType"] == 'MultipleChoice')])

result_dict = {}
for i in MCcols:
    # Check to see if an object 
    if isinstance(Results[i], object):
        NewList = []
        # then need to go through and extract the data
        for ii in Results[i]:
            if isinstance(ii, list):
                # Go through extract the list
                NewList = NewList + ii
            elif isinstance(ii, float):
                # Check to see if nan
                if np.isnan(ii):
                    # add a null
                    NewList = NewList + ['NULL']
                else:
                    # Convert to integer and string
                    NewList = NewList + [str(int(ii))]
            else:
                print('Problems with conversion')
    elif Results[i].dtypes == 'float':
        # Then 
        NewList = []
        for ii in Results[i]:
            if np.isnan(ii):
                # add a null
                NewList = NewList + ['NULL']
            else:
                # Convert to integer and string
                NewList = NewList + [str(int(ii))]
    else:
        print('Problems with conversion')
                
    Fdist = Counter(NewList)
    freqDF = pd.DataFrame.from_dict(Fdist, orient='index').reset_index()
    freqDF.columns = ['Q_AnsId', 'N']    
    # Create a new DF with the Q_AnsId, Q_Value, N, Pct
    CurVals = QVals.loc[np.array(QVals['QID']) == i, ['Q_AnsId', 'Q_Value']]
    tdf = pd.merge(CurVals, freqDF, on = 'Q_AnsId', how = 'outer')
    # Replace nans with zero
    Mask = np.array(np.isnan(tdf['N']))
    tdf.loc[Mask, 'N'] = 0
    tdf['N'] = tdf['N'].astype('int')
    # Replace nans with ''
    Mask = np.array(tdf['Q_Value'].isna())
    tdf.loc[Mask, 'Q_Value'] = ''
    # Get the percent
    tdf['Pct'] = tdf['N'] / len(Results) * 100
    tdf['Pct'] = tdf['Pct'].map('{0:.1f}'.format) + '%'
    
    # Set the names of the columns to however you want
    tdf.columns = ['Code', 'Value', 'Count', 'Frequency']
    # Code to calculate frequencies
    result_dict[i] = tdf
    
# Print the results from the dictionary
for column, result_df in result_dict.items():
    print(f"Column: {column}")
    print(result_df)
    print()


# Save frequencies as a word document
# Create a new Word document
# To plot only the values for regions that werent 0. list the questions below
regional_questions = ['Please select your SOAF Program: ', 'Please select your SOAP Program: ',
'Please select your SOEA Program:', 'Please select your SOEE Program:',
'Please select your SOLA Program:', 'Please select your MENA Program:',
'Please select your SONA Program:']


doc = Document()

section = doc.sections[0]

# Set the margins (all to 1 inch in this example)
section.top_margin = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin = Inches(1)
section.right_margin = Inches(1)

# Iterate through each column's results and add a table and a bar chart to the document
for original_column, result_df in result_dict.items():
    # Find the corresponding QText in QDF for the current QID
    qname = QDF[QDF['QID'] == original_column]['QName'].values[0]
    qtext = QDF[QDF['QID'] == original_column]['QText'].values[0]
    regional_questions = ['Please select your SOAF Program: ', 'Please select your SOAP Program: ',
    'Please select your SOEA Program:', 'Please select your SOEE Program:',
    'Please select your SOLA Program:', 'Please select your MENA Program:',
    'Please select your SONA Program:']
    if qtext in regional_questions:
        # Apply the filter
        filtered_result_df = result_df[(result_df['Count'] != 0)]
    else:
        filtered_result_df = result_df

    # Combine QName and QText with a separator like ' - ' or ': '
    mapped_text = f"{qname}: {qtext}"
    
    # Add mapped_text as a heading
    doc.add_heading(mapped_text, level=1)
    
    # Add a table
    table = doc.add_table(rows=result_df.shape[0] + 1, cols=result_df.shape[1])
    table.style = 'Table Grid'
    
    column_alignment = [1] * result_df.shape[1]  # Initialize all columns to center alignment
    value_column_index = result_df.columns.get_loc("Value")
    column_alignment[value_column_index] = 0  # Set 'Value' column to left alignment

    for row in table.rows:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.alignment = 1  # Center alignment
            cell.vertical_alignment = 1  # Center vertically
            cell.width = Inches(2)
           
    for col_idx, column_name in enumerate(result_df.columns):
        cell = table.cell(0, col_idx)
        cell.text = column_name
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True  # Make the text bold
                run.font.size = Pt(14)  # Adjust font size if needed
            paragraph.alignment = 1  # Center alignment
    
    for row_idx in range(result_df.shape[0]):
        for col_idx, value in enumerate(result_df.iloc[row_idx]):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(value)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(12)  # Adjust font size if needed
            cell.paragraphs[0].alignment = column_alignment[col_idx]

    # Code below to not make plots if there are more than 10 rows in result_df  
    # if len(result_df) >= 10:
    #     continue
    # Create a new figure and axes for the bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_width = 0.4

    #frequency_values = result_df['Frequency'].astype(str).str.rstrip('%').astype(float)
    frequency_values = filtered_result_df['Frequency'].astype(str).str.rstrip('%').astype(float)

    # Plotting the bar chart with the Frequency percentage
    usual_bar_color = 'dodgerblue'
    null_bar_color = 'crimson'

    # Plot the bars, keeping the colors in a list, which will be the same length as the number of bars.
    bar_colors = [usual_bar_color if code != 'NULL' else null_bar_color for code in filtered_result_df['Code']]
    
    ax.bar(filtered_result_df['Code'], frequency_values, color=bar_colors, width=bar_width)
    ax.set_xlabel('Code')
    ax.set_ylabel('Frequency (%)')
    ax.set_xticks(filtered_result_df['Code'])

    ax.set_ylim(0, 100) 
  # Set y-axis scale from 0 to 100%
  # Bar labels as count:
    # for p, count in zip(ax.patches, filtered_result_df['Count']):
    #     ax.annotate(
    #         str(count),  # The label text, which is the count value
    #         (p.get_x() + p.get_width() / 2., p.get_height()),  # The position (x,y)
    #         ha='center',  # Center alignment of text
    #         va='bottom',  # Align the bottom of the text at the given position
    #         xytext=(0, 10),  # 10 points vertical offset
    #         textcoords='offset points',  # Offset (in points) from the specified position
    #         fontsize=10,  # Font size of the text
    #     )
    # Bar labels at frequency:
    for p, freq in zip(ax.patches, filtered_result_df['Frequency']):
        height = p.get_height()
        ax.annotate('{}'.format(freq),
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Save the bar chart as an image
    chart_filename = f'bar_chart_{original_column}.png'
    plt.savefig(chart_filename)
    plt.close(fig)  # Close the figure
    doc.add_picture(chart_filename, width=Inches(7), height=Inches(4.5))
    doc.add_page_break()

# Save the Word document
#doc.save('/Users/kieranmartin/Documents/SOI Data/Python/Qualtrics_API_Program/Reports/lifestle3.docx')
doc.save('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics_API_Program\\Reports\\test.docx')


# # Create bar charts using the dictionary created above
# # Generate bar charts for each frequency distribution
# for column, result_df in result_dict.items():
#     # Convert the 'Count' column to integers for plotting
#     result_df['Count'] = result_df['Count'].astype(int)
    
#     # Create a bar chart
#     plt.figure(figsize=(10, 6))
#     plt.bar(result_df['Code'], result_df['Count'])
    
#     # Add labels and title
#     plt.xlabel('Code')
#     plt.ylabel('Count')
#     plt.title(f'Frequency Distribution of {column}')
#     plt.xticks(result_df['Code'])
    
#     # Display the percentage on top of each bar
#     for index, row in result_df.iterrows():
#         label_y = row['Count'] + 5  
#         plt.text(row['Code'], label_y, f"{row['Code']} ({row['Percentage']})", ha='center', va='bottom')
    
#     # Show the plot
#     plt.tight_layout()
#     plt.show()




