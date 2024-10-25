# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:37:04 2024

@author: 484843
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
survey_name = "Spring 2024 COPA People's Academy: Post-Participation Survey"

############################################################################### 
base_url = 'https://{0}.qualtrics.com'.format(data_center)

# set up the data for the token
grant_type = 'client_credentials'
scope = 'read:surveys read:survey_responses'
data = qa.return_kwargs_as_dict(grant_type = grant_type, scope = scope)

# Get the token and extract the Bearer access toke
bearer_token_response = qa.get_token(base_url, client_id, client_secret, data)
token = bearer_token_response.get("access_token")

# Pull the list of surveys
survey_list_df = qa.get_survey_list(base_url, token)

def get_survey_id_by_name(survey_list_df, survey_name):
    """
    Retrieves the survey ID based on the survey name from a DataFrame.

    Args:
        survey_list_df (pd.DataFrame): A DataFrame containing survey data with 'name' and 'id' columns.
        survey_name (str): The name of the survey to search for.

    Returns:
        str: The survey ID if found, or None if not found or multiple surveys have the same name.
    """
    # Pull the indices associated with the Survey Name
    survey_name_indices = survey_list_df.loc[survey_list_df['name'] == survey_name].index[:]
    
    if len(survey_name_indices) > 1:
        print('Multiple Surveys Have the Same Name!!!')
        return None  # Return None to indicate multiple surveys found
    elif len(survey_name_indices) == 0:
        print('Cannot Find the Survey. Please Check that the Survey Name is Correct.')
        return None  # Return None to indicate survey not found
    else:
        # Return the survey ID using the survey index
        survey_id = survey_list_df.loc[survey_name_indices[0], 'id']
        return survey_id

survey_id = get_survey_id_by_name(survey_list_df, survey_name)


# Get the Response Export
response_export = qa.export_survey_responses(base_url, token, survey_id)
# Get the exportProgressId
export_progress_id = response_export.get('result').get('progressId')

def wait_for_export_completion(base_url, token, survey_id, export_progress_id):
    """
    Waits for the survey response export to complete by checking its progress.

    Args:
        base_url (str): The base URL for the Qualtrics API.
        token (str): The API token used for authorization.
        survey_id (str): The ID of the survey being exported.
        export_progress_id (str): The progress ID of the ongoing export.

    Returns:
        str: The file ID (fid) when the export is complete.
    """
    # Initialize the file ID to empty
    fid = ""
    
    # Loop until the export is complete and the file ID is available
    while len(fid) == 0:
        # Check the progress of the export
        response_export_progress = qa.get_response_export_progress(base_url, token, survey_id, export_progress_id)

        # Check if the export is 100% complete
        if response_export_progress.get('result').get('percentComplete') == 100:
            # Check if the status is 'complete'
            if response_export_progress.get('result').get('status') == 'complete':
                # Get the file ID
                fid = response_export_progress.get('result').get('fileId')
            else:
                print("Status: " + response_export_progress.get('result').get('status'))
    
    return fid

# Wait for the export to complete and retrieve the file ID
fid = wait_for_export_completion(base_url, token, survey_id, export_progress_id)

# Get the survey responses
survey_responses = qa.get_survey_responses(base_url, token, survey_id, fid)


def extract_and_organize_responses(survey_responses):
    """
    Extracts and organizes survey responses from the JSON response.

    Args:
        survey_responses (requests.Response): The response object from the survey responses export.

    Returns:
        pd.DataFrame or None: A DataFrame containing the organized responses, or None if no responses are available.
    """
    # Convert the response content to JSON
    survey_responses_json = survey_responses.json()
    
    # Extract the list of responses
    responses = survey_responses_json.get('responses', [])
    
    if len(responses) > 0:
        # Organize the responses using qa's organize_responses function
        responses_df = qa.organize_responses(responses)
        return responses_df
    else:
        print('There is no data')
        return None

responses_df = extract_and_organize_responses(survey_responses)

def filter_preview_responses(responses_df):
    """
    Filters out survey preview responses from the responses DataFrame and resets the index.
    
    Parameters:
    responses_df (pd.DataFrame): DataFrame containing survey responses with a 'distributionChannel' column.
    
    Returns:
    pd.DataFrame: Filtered DataFrame with preview responses removed and index reset.
    """
    # Do not keep survey preview responses
    keep_mask = np.array(responses_df['distributionChannel'] != 'preview')
    filtered_df = responses_df.loc[keep_mask, :].reset_index(drop=True)
    
    return filtered_df

# Call the function
responses_df = filter_preview_responses(responses_df)


# Get the survey questions
surveyq_dict = qa.get_survey_questions(base_url, token, survey_id)

# Map the Survey Qs so that they can match with column headers of Results
survey_questions_result = surveyq_dict.get('result')
question_dictionary = survey_questions_result.get('questions')

# Get the column data types
question_df, question_values_df = qa.extract_column_data_types(question_dictionary, responses_df, base_url, token, survey_id)

# Get the different data types for reporting purposes
question_df = qa.create_data_type_dictionary(question_df, question_values_df)


def clean_responses(responses_df, question_df):
    """
    Extracts columns with questions from responses_df, removes rows with all NaN values 
    in those columns, and resets the index.
    
    Parameters:
    responses_df (pd.DataFrame): DataFrame containing survey responses.
    question_df (pd.DataFrame): DataFrame containing questions, with a 'question_id' column.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with rows containing all NaN values removed and index reset.
    """
    # Extract only columns with questions
    df = responses_df.loc[:, question_df['question_id'].tolist()]

    # Find rows with all NaN values in question columns
    nan_mask = df.isna()
    keep_mask = np.array(nan_mask.sum(axis=1) < len(df.columns))

    # Filter responses_df based on keep_mask and reset index
    responses_df = responses_df.loc[keep_mask, :].reset_index(drop=True)
    
    return responses_df

# Call the function
responses_df = clean_responses(responses_df, question_df)

###############################################################################
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

# df = subset_by_date_range(df, '2023-01-01', '2023-03-31')
def process_response_column(responses_df, column):
    """
    Cleans and processes a multiple-choice response column by checking types 
    and handling lists, NaNs, and other cases as in the original structure.
    
    Parameters:
    responses_df (pd.DataFrame): Survey responses DataFrame.
    column (str): Column name for the multiple-choice question.
    
    Returns:
    list: Processed list of responses with 'NULL' for NaNs.
    """
    new_list = []
    
    # Check if the column is of object type
    if isinstance(responses_df[column], object):
        for response in responses_df[column]:
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
                
    elif responses_df[column].dtype == 'float':
        for response in responses_df[column]:
            if np.isnan(response):
                new_list.append('NULL')
            else:
                new_list.append(str(int(response)))
    else:
        print('Problems with conversion')
    
    return new_list



def generate_response_frequency(responses_df, question_df, question_values_df):
    """
    Generates frequency tables and plots for multiple-choice questions without regional question filtering,
    and saves them to a Word document.
    
    Parameters:
    responses_df (pd.DataFrame): Survey responses DataFrame.
    question_df (pd.DataFrame): DataFrame with question details, including 'question_id' and 'data_type'.
    question_values_df (pd.DataFrame): DataFrame with answer IDs and question values.

    Returns:
    Document: A Word document with tables and bar charts for each question.
    """
    result_dict = {}
    multiple_choice_columns = list(question_df.question_id[question_df["data_type"] == 'MultipleChoice'])

    for col in multiple_choice_columns:
        new_list = process_response_column(responses_df, col)
        freq_dist = Counter(new_list)
        freq_df = pd.DataFrame.from_dict(freq_dist, orient='index').reset_index()
        freq_df.columns = ['answer_id', 'N']
        
        current_values = question_values_df[question_values_df['question_id'] == col][['answer_id', 'question_value']]
        temp_df = pd.merge(current_values, freq_df, on='answer_id', how='outer').fillna({'N': 0, 'question_value': ''})
        temp_df['N'] = temp_df['N'].astype(int)
        temp_df['Pct'] = (temp_df['N'] / len(responses_df) * 100).round(1).astype(str) + '%'
        temp_df.columns = ['Code', 'Value', 'Count', 'Frequency']
        
        result_dict[col] = temp_df
        # Print the frequency table for the current question
        # print(f"Frequency table for question {col}:\n{temp_df}\n")

    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)

    for original_column, result_df in result_dict.items():
        qname = question_df.loc[question_df['question_id'] == original_column, 'question_name'].values[0]
        qtext = question_df.loc[question_df['question_id'] == original_column, 'question_text'].values[0]
        mapped_text = f"{qname}: {qtext}"
        
        doc.add_heading(mapped_text, level=1)
        filtered_result_df = result_df

        table = doc.add_table(rows=filtered_result_df.shape[0] + 1, cols=filtered_result_df.shape[1])
        table.style = 'Table Grid'
        
        for col_idx, column_name in enumerate(filtered_result_df.columns):
            cell = table.cell(0, col_idx)
            cell.text = column_name
            cell.paragraphs[0].alignment = 1
        
        for row_idx in range(filtered_result_df.shape[0]):
            for col_idx, value in enumerate(filtered_result_df.iloc[row_idx]):
                cell = table.cell(row_idx + 1, col_idx)
                cell.text = str(value)
                cell.paragraphs[0].alignment = 1 if col_idx != 1 else 0  # Align 'Value' left, others center

        if len(filtered_result_df) < 10:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            frequency_values = filtered_result_df['Frequency'].str.rstrip('%').astype(float)
            bar_colors = ['dodgerblue' if code != 'NULL' else 'crimson' for code in filtered_result_df['Code']]
            ax.bar(filtered_result_df['Code'], frequency_values, color=bar_colors, width=0.4)
            ax.set_xlabel('Code')
            ax.set_ylabel('Frequency (%)')
            ax.set_xticks(filtered_result_df['Code'])
            ax.set_ylim(0, 100)

            for p, freq in zip(ax.patches, filtered_result_df['Frequency']):
                ax.annotate(f'{freq}', (p.get_x() + p.get_width() / 2, p.get_height() + 3), ha='center', fontsize=10)

            image_stream = BytesIO()
            plt.savefig(image_stream, format='png')
            plt.close(fig)
            image_stream.seek(0)
            doc.add_picture(image_stream, width=Inches(7), height=Inches(4.5))
            
            # If you WANT to save the indivual plots use code below instead:
            # Save the bar chart as an image
            # chart_filename = f'bar_chart_{original_column}.png'
            # plt.savefig(chart_filename)
            # plt.close(fig)  # Close the figure
            # doc.add_picture(chart_filename, width=Inches(7), height=Inches(4.5))

        doc.add_page_break()

    return doc

doc = generate_response_frequency(responses_df, question_df, question_values_df)
doc.save('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics_API_Program\\Reports\\test14.docx')

# Save the document



# # For each multiple choice question create a table with the frequency responses
## and fractions
# Extract Multiple Choice columns
# multiple_choice_columns = list(question_df.question_id[np.array(question_df["data_type"] == 'MultipleChoice')])

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




# # Save frequencies as a word document
# # Create a new Word document
# # To plot only the values for regions that werent 0. list the questions below
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
    # Create a buffer to hold the image
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
doc.save('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics_API_Program\\Reports\\test13.docx')






