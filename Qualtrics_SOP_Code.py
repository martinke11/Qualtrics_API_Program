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
from io import BytesIO
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
    
# Extract the clientId, client Secret, and dataCenter
client_id = creds.get('ID')
client_secret = creds.get('Secret')
data_center = creds.get('DataCenter')

# Extract the Survey ID based on the name of the Survey
# survey_name = 'Community Health Worker (CHW) Training: Pre-Training Survey'
survey_name = "Spring 2024 COPA People's Academy: Post-Participation Survey"
# survey_name = 'Current Y2 HIG Health Evaluation Report 2022-23'


###############################################################################
# create the base url 
base_url = 'https://{0}.qualtrics.com'.format(data_center)

# set up the data for the token
grant_type = 'client_credentials'
#scope = 'read:surveys read:survey_responses manage:all manage:users read:users write:users'
scope = 'read:surveys read:survey_responses'
data = qa.return_kwargs_as_dict(grant_type = grant_type, scope = scope)

# Get the token
bearer_token_response = qa.get_token(base_url, client_id, client_secret, data)
# Extract the Bearer access token
token = bearer_token_response.get("access_token")

# Pull the list of surveys
survey_list_df = qa.get_survey_list(base_url, token)

# Pull the indices associated with the Survey Name
survey_name_indices = survey_list_df.loc[survey_list_df['name'] == survey_name].index[:]

if len(survey_name_indices) > 1:
    print('Multiple Surveys Have the Same Name!!!')
elif len(survey_name_indices) == 0:
    print('Cannot Find the Survey. Please Check that the Survey Name is Correct.')
elif len(survey_name_indices) == 1:
    survey_name_indices = survey_name_indices[0]    

# Extract the Survey Id 
survey_id = survey_list_df.loc[survey_name_indices, 'id']

# Get the Response Export
response_export = qa.export_survey_responses(base_url, token, survey_id)
# Get the exportProgressId
export_progress_id = response_export.get('result').get('progressId')

# wait until the status is 100% complete
# set fid to blank
fid = ""
while len(fid) == 0:    
    # Check the progress 
    response_export_progress = qa.get_response_export_progress(base_url, token, survey_id, export_progress_id)

    # # Check if the percent complete is 100
    if response_export_progress.get('result').get('percentComplete') == 100:
        # Then check to see if the pull is complete
        if response_export_progress.get('result').get('status') == 'complete':
            # Get the file Id
            fid = response_export_progress.get('result').get('fileId')
        else:
            print("Status: " + response_export_progress.get('result').get('status'))

# Get the survey responses
survey_responses = qa.get_survey_responses(base_url, token, survey_id, fid)
# Extract and arrange data
# Convert to JSON
survey_responses_json = survey_responses.json()
# Get the list of responses
responses = survey_responses_json.get('responses')

if len(responses) > 0:
    responses_df = qa.organize_responses(responses)
else:
    print('There is no data')

print(responses)
###############################################################################
# do not keep survey preview  responses
keep_mask = np.array(responses_df['distributionChannel'] != 'preview')
responses_df = responses_df.loc[keep_mask, :]
# Reset the index
responses_df = responses_df.reset_index(drop = True)
###############################################################################
# organized_responses_df = organized_responses_df[organized_responses_df['progress'] >= 100]
# organized_responses_df = organized_responses_df[organized_responses_df['QID3'] == 2]

# Get the survey questions
surveyq_dict = qa.get_survey_questions(base_url, token, survey_id)

# Map the Survey Qs so that they can match with column headers of Results
survey_questions_result = surveyq_dict.get('result')

question_dictionary = survey_questions_result.get('questions')

# Get the column data types
question_df, question_values_df = qa.extract_column_data_types(question_dictionary, responses_df, base_url, token, survey_id)

# Get the different data types for reporting purposes
question_df = qa.create_data_type_dictionary(question_df, question_values_df)

print(question_df.columns)

###############################################################################
# Using the question_df, extract only those columns with questions
df = responses_df.loc[:, question_df['question_id'].tolist()]

# This code will find all the NaN values. It can be used to clean and determine NaN frequencies
nan_mask = df.isna()
keep_mask = np.array(nan_mask.sum(axis=1) < len(df.columns))

# Now adjust responses_df and df based on keep_mask
responses_df = responses_df.loc[keep_mask, :]
df = df.loc[keep_mask, :]

# Reset the index for both DataFrames
responses_df = responses_df.reset_index(drop=True)
df = df.reset_index(drop=True)


###############################################################################
nan_mask = df.isna()
missing_data = nan_mask.sum(axis=0).tolist()
present_data = len(df) - np.array(missing_data)

# Get the fraction of respondents by question
missing_data_fraction = np.array(missing_data) / len(df)
present_data_fraction = 1 - missing_data_fraction

# Create a DataFrame
missing_text_df = {
    'question_id': df.columns,
    "MissingDataN": missing_data,
    "PresentDataN": present_data,
    "MissingDataFrc": missing_data_fraction,
    "PresentDataFrc": present_data_fraction
}

missing_text_df = pd.DataFrame(missing_text_df)

free_text_columns = question_df.question_id[np.array(question_df.data_type == 'FreeText')]

# Grab the missing data for the text
mask_missing_text_df = np.array(np.isin(missing_text_df['question_id'], list(free_text_columns)))
mask_missing_text_df = missing_text_df.loc[mask_missing_text_df, :]


# Inquire about the date range of the data to pull and delete any data
# that falls outside the date range
# Have start and end date in the csv file name. Example: data_startdate_enddate.csv
# Change date column 'RecordedDate' to date format
# def subset_by_date_range(df, start_date, end_date):
#     # Ensure the 'recorded_date' column is in datetime format
#     df['recorded_date'] = pd.to_datetime(df['recorded_date'])

#     # Convert the start and end dates to datetime
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)

#     # Subset the DataFrame based on the date range
#     subset_df = df[(df['recorded_date'] >= start_date) & (df['recorded_date'] <= end_date)]

#     return subset_df

# # Example usage:
# df = subset_by_date_range(df, '2023-01-01', '2023-03-31')
print(question_values_df.columns)



# For each multiple choice question create a table with the frequency responses
# and fractions
# Extract Multiple Choice columns
# Extract Multiple Choice columns
multiple_choice_columns = list(question_df.question_id[np.array(question_df["data_type"] == 'MultipleChoice')])

result_dict = {}
for col in multiple_choice_columns:
    # Check if the column is an object type
    if isinstance(responses_df[col], object):
        new_list = []
        # Loop through the responses in the column
        for response in responses_df[col]:
            if isinstance(response, list):
                # Add the list of responses
                new_list += response
            elif isinstance(response, float):
                # Check if it's NaN and handle accordingly
                if np.isnan(response):
                    new_list.append('NULL')
                else:
                    new_list.append(str(int(response)))
            else:
                print('Problems with conversion')
    elif responses_df[col].dtypes == 'float':
        new_list = []
        for response in responses_df[col]:
            if np.isnan(response):
                new_list.append('NULL')
            else:
                new_list.append(str(int(response)))
    else:
        print('Problems with conversion')
                
    # Create a frequency distribution of the new list
    freq_dist = Counter(new_list)
    freq_df = pd.DataFrame.from_dict(freq_dist, orient='index').reset_index()
    freq_df.columns = ['answer_id', 'N']  # Updated based on the column name in question_values_df
    
    # Create a new DataFrame with question_id, question_value, N, Pct
    current_values = question_values_df.loc[np.array(question_values_df['question_id']) == col, ['answer_id', 'question_value']]
    temp_df = pd.merge(current_values, freq_df, on='answer_id', how='outer')

    # Replace NaNs with zero for counts
    mask = np.array(np.isnan(temp_df['N']))
    temp_df.loc[mask, 'N'] = 0
    temp_df['N'] = temp_df['N'].astype('int')

    # Replace NaNs with empty strings for question values
    mask = np.array(temp_df['question_value'].isna())
    temp_df.loc[mask, 'question_value'] = ''

    # Calculate the percentage
    temp_df['Pct'] = (temp_df['N'] / len(responses_df)) * 100
    temp_df['Pct'] = temp_df['Pct'].map('{0:.1f}'.format) + '%'
    
    # Set the column names to desired labels
    temp_df.columns = ['Code', 'Value', 'Count', 'Frequency']
    
    # Store the resulting DataFrame in the dictionary
    result_dict[col] = temp_df
    
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
    # Find the corresponding question_name and question_text in question_df for the current question_id
    qname = question_df[question_df['question_id'] == original_column]['question_name'].values[0]
    qtext = question_df[question_df['question_id'] == original_column]['question_text'].values[0]
    
    if qtext in regional_questions:
        # Apply the filter
        filtered_result_df = result_df[(result_df['Count'] != 0)]
    else:
        filtered_result_df = result_df

    # Combine question_name and question_text with a separator like ' - ' or ': '
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

    # Skip creating plots if there are more than 10 rows in result_df
    if len(result_df) >= 10:
        continue

    # Create a new figure and axes for the bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_width = 0.4

    # Extract frequency values for the bar chart
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
  # To make Count the Bar labels use this code:
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
    # Bar labels as frequency:
    for p, freq in zip(ax.patches, filtered_result_df['Frequency']):
        height = p.get_height()
        ax.annotate('{}'.format(freq),
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


    # In order to not save the bar plots as indivuals picture files in folder 
    # use image stream:
    # Create a buffer to hold the imag
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close(fig)

    # Move the pointer in the buffer to the start
    image_stream.seek(0)
    
    # Add the image from the buffer to the Word document
    doc.add_picture(image_stream, width=Inches(7), height=Inches(4.5))
    doc.add_page_break()
    
    # If you WANT to save the indivual plots use code below instead:
    # Save the bar chart as an image
    # chart_filename = f'bar_chart_{original_column}.png'
    # plt.savefig(chart_filename)
    # plt.close(fig)  # Close the figure
    # doc.add_picture(chart_filename, width=Inches(7), height=Inches(4.5))
    # doc.add_page_break()

# Save the Word document
#doc.save('/Users/kieranmartin/Documents/SOI Data/Python/Qualtrics_API_Program/Reports/lifestle3.docx')
doc.save('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics_API_Program\\Reports\\test11.docx')







