# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:50:53 2023

@author: kmartin
"""

# Qualtrics API Functions
# Reference for Qualtrics API endpoints:
# https://api.qualtrics.com/0f8fac59d1995-api-reference

import requests
import pandas as pd
import numpy as np
import re

def convert_kwargs_to_string(**kwargs):
    """
    Converts the given keyword arguments into a string representation.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        str: A string representation of the keyword arguments.
    """
    kwargs_string = str(kwargs)
    return kwargs_string


def format_dict_for_json(dictionary):
    """
    Formats a dictionary by converting it to a JSON-like string.
    Replaces single quotes with double quotes for proper JSON formatting.

    Args:
        dictionary (dict): A dictionary to format.

    Returns:
        str: A JSON-like formatted string with double quotes.
    """
    json_like_string = str(dictionary).replace("'", '"')
    return json_like_string


def return_kwargs_as_dict(**kwargs):
    """
    Returns the keyword arguments as a dictionary.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        dict: A dictionary containing the keyword arguments.
    """
    kwargs_dict = kwargs
    return kwargs_dict


def get_token(base_url, client_id, client_secret, data):
    """
   Requests an OAuth2 token that will last for one hour.

   Args:
       base_url (str): The base URL for the API.
       client_id (str): The client ID for authentication.
       client_secret (str): The client secret for authentication.
       data (dict): A dictionary containing data for the token request, including
                    grant type and other relevant parameters.

   Returns:
       dict: A JSON object containing the token and related information.

   Available scopes:
       - manage:activity_logs
       - manage:all
       - manage:contact_frequency_rules
       - manage:contact_transactions
       - manage:customer_data_requests
       - manage:directories
       - manage:directory_contacts
       - manage:distributions
       - manage:divisions
       - manage:erasure_requests
       - manage:groups
       - manage:libraries
       - manage:mailing_list_contacts
       - manage:mailing_lists
       - manage:organizations
       - manage:participants
       - manage:samples
       - manage:subscriptions
       - manage:survey_responses
       - manage:survey_sessions
       - manage:surveys
       - read:activity_logs
       - read:contact_frequency_rules
       - read:contact_transactions
       - read:directories
       - read:directory_contacts
       - read:distributions
       - read:divisions
       - read:groups
       - read:imported_data_projects
       - read:libraries
       - read:mailing_list_contacts
       - read:mailing_lists
       - read:organizations
       - read:participants
       - read:samples
       - read:subscriptions
       - read:survey_responses
       - read:survey_sessions
       - read:surveys
       - read:users
       - write:automations
       - write:contact_frequency_rules
       - write:contact_transactions
       - write:directory_contacts
       - write:distributions
       - write:divisions
       - write:embedded_dashboards
       - write:embedded_xid_profile_cards
       - write:groups
       - write:imported_data_projects
       - write:libraries
       - write:mailing_list_contacts
       - write:mailing_lists
       - write:participants
       - write:samples
       - write:subscriptions
       - write:survey_responses
       - write:survey_sessions
       - write:surveys
   """
    manage:activity_logs
    manage:all
    manage:contact_frequency_rules
    manage:contact_transactions
    manage:customer_data_requests
    manage:directories
    manage:directory_contacts
    manage:distributions
    manage:divisions
    manage:erasure_requests
    manage:groups
    manage:libraries
    manage:mailing_list_contacts
    manage:mailing_lists
    manage:organizations
    manage:participants
    manage:samples
    manage:subscriptions
    manage:survey_responses
    manage:survey_sessions
    manage:surveys
    read:activity_logs
    read:contact_frequency_rules
    read:contact_transactions
    read:directories
    read:directory_contacts
    read:distributions
    read:divisions
    read:groups
    read:imported_data_projects
    read:libraries
    read:mailing_list_contacts
    read:mailing_lists
    read:organizations
    read:participants
    read:samples
    read:subscriptions
    read:survey_responses
    read:survey_sessions
    read:users
    read:surveys
    write:automations
    write:contact_frequency_rules
    write:contact_transactions
    write:directory_contacts
    write:distributions
    write:divisions
    write:embedded_dashboards
    write:embedded_xid_profile_cards
    write:groups
    write:imported_data_projects
    write:libraries
    write:mailing_list_contacts
    write:mailing_lists
    write:participants
    write:samples
    write:subscriptions
    write:survey_responses
    write:survey_sessions
    write:surveys
    write:users
    manage:subscriptions
    manage:survey_responses
    manage:survey_sessions
    manage:surveys
    read:subscriptions
    read:survey_responses
    read:survey_sessions
    read:surveys
    write:subscriptions
    write:survey_responses
    write:survey_sessions
    write:surveys

    token_url = base_url + "/oauth2/token"
    
    response = requests.post(token_url, auth=(client_id, client_secret), data=data)
    
    return response.json()

# ========================================================
# Survey Handling Functions
# ========================================================
def get_survey_list(base_url, token):
    """
    Retrieve a complete list of surveys from the Qualtrics API, handling pagination if necessary.
    
    This function makes repeated API requests to the Qualtrics survey endpoint to retrieve all surveys.
    It handles the pagination of results by checking for a "nextPage" key in the API response, which 
    provides an offset for subsequent pages of data. The function returns the complete survey list in 
    the form of a Pandas DataFrame.
    
    Args:
        base_url (str): The base URL for the Qualtrics API.
        token (str): The API token used for authorization.

    Returns:
        pd.DataFrame: A DataFrame containing the complete list of surveys with their details.
        
    Raises:
        requests.exceptions.RequestException: If there's an error with the API request.
    """
    # Initialize the flag to track pagination and the offset for next pages
    flag = True
    count = 0
    offset = ""
    
    while flag:
        # Determine the URL based on whether it's the first page or a subsequent page
        if count == 0:
            url = '{0}/API/v3/surveys'.format(base_url)
        else:
            url = '{0}/API/v3/surveys?offset={1}'.format(base_url, offset)
        
        # Make the API request
        response = requests.get(url, headers={"Authorization": "Bearer " + token})
        response_json = response.json()

        # Extract the current page of survey results into a DataFrame
        current_page_df = pd.DataFrame(response_json.get("result").get("elements"))
        
        # For the first page, initialize the DataFrame; for others, append to it
        if count == 0:
            survey_list_df = current_page_df
        else:
            survey_list_df = pd.concat([survey_list_df, current_page_df])
            
        # Check if there's a next page and update flag and offset accordingly
        if response_json.get("result").get("nextPage") is None:
            flag = False
        else:
            next_page_str = response_json.get("result").get("nextPage")
            offset = next_page_str.split("?offset=", 1)[1]
            count += 1
    
    # Reset the indices of the final DataFrame and return it
    survey_list_df = survey_list_df.reset_index(drop=True)
    
    return survey_list_df


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


# ========================================================
# Survey Response Handling Functions
# ========================================================
def export_survey_responses(base_url, access_token, survey_id):
    """
    Initiates the export of survey responses from Qualtrics.

    Args:
        base_url (str): The base URL for the Qualtrics API.
        access_token (str): The bearer access token for API authorization.
        survey_id (str): The unique ID of the survey whose responses are to be exported.

    Returns:
        dict: A JSON-formatted response containing the export details, such as progress and file ID.
    """
    # Set the survey export URL
    survey_export_url = '{0}/API/v3/surveys/{1}/export-responses'.format(base_url, survey_id)
    # Pull the survey data
    response = requests.post(survey_export_url, 
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + access_token},
                             data='{"format": "json", "compress": false}')
    # Convert the data into a more readable format
    response = response.json()    
    return response


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
        response_export_progress = get_response_export_progress(base_url, token, survey_id, export_progress_id)

        # Check if the export is 100% complete
        if response_export_progress.get('result').get('percentComplete') == 100:
            # Check if the status is 'complete'
            if response_export_progress.get('result').get('status') == 'complete':
                # Get the file ID
                fid = response_export_progress.get('result').get('fileId')
            else:
                print("Status: " + response_export_progress.get('result').get('status'))
    
    return fid


def get_response_export_progress(base_url, access_token, survey_id, export_progress_id):
    """
    Checks the progress of an ongoing survey response export.

    Args:
        base_url (str): The base URL for the Qualtrics API.
        access_token (str): The bearer access token for API authorization.
        survey_id (str): The unique ID of the survey whose responses are being exported.
        export_progress_id (str): The unique ID representing the current export progress.

    Returns:
        dict: A JSON-formatted response indicating the progress status and completion percentage.
    """
    # Set the export progress URL
    export_progress_url = '{0}/API/v3/surveys/{1}/export-responses/{2}'.format(base_url, survey_id, export_progress_id)
    # Pull the survey data
    response = requests.get(export_progress_url, 
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + access_token})
    # Convert the data into a more readable format
    response = response.json()    
    return response


def get_survey_responses(base_url, access_token, survey_id, file_id):
    """
    Downloads the survey responses after the export process is complete.

    Args:
        base_url (str): The base URL for the Qualtrics API.
        access_token (str): The bearer access token for API authorization.
        survey_id (str): The unique ID of the survey whose responses are to be downloaded.
        file_id (str): The file ID representing the exported survey responses.

    Returns:
        requests.Response: The HTTP response containing the survey responses file.
    """
    # Set the file download URL
    file_download_url = '{0}/API/v3/surveys/{1}/export-responses/{2}/file'.format(base_url, survey_id, file_id)
    # Pull the survey data
    response = requests.get(file_download_url, 
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + access_token})
    return response


#below is a moved function:
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
        responses_df = organize_responses(responses)
        return responses_df
    else:
        print('There is no data')
        return None
    

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


def organize_responses(responses):
    """
    Organize survey responses into a structured DataFrame.
    
    Args:
        responses (list): List of survey response data in JSON format.
    
    Returns:
        pd.DataFrame: A DataFrame containing the organized responses.
    """
    # Initialize an empty DataFrame for the responses
    responses_df = []

    # Process each response
    for response in responses:
        # Extract the current response and normalize it into a DataFrame
        temp_df = pd.json_normalize(response.get('values'))
        if len(responses_df) == 0:
            responses_df = temp_df
        else:
            # Concatenate the responses into the DataFrame
            responses_df = pd.concat([responses_df, temp_df])
    
    # Reset the index of the DataFrame
    responses_df = responses_df.reset_index(drop=True)
    
    # Reorder the columns: first non-question columns, then question columns
    column_names = responses_df.columns
    question_columns = [col for col in column_names if col.startswith('QID')]
    non_question_columns = list(set(column_names) - set(question_columns))
    
    non_question_df = responses_df.loc[:, np.isin(column_names, non_question_columns)]

    # Prepare to order question columns by number, sub-question, and loop number
    question_number = []
    sub_question_number = []
    loop_number = []

    for question in question_columns:
        # Split the column name into parts (e.g., 'QID1_1_TEXT' becomes ['QID1', '1', 'TEXT'])
        split_column_name = re.split('_|#', question)
        
        # Find the index of the 'QID' part of the split
        qid_index = [i for i in range(len(split_column_name)) if 'QID' in split_column_name[i]]
        max_index = len(split_column_name) - 1

        # Determine if it's a loop question
        if qid_index[0] == 0:
            loop_number.append(0)
        elif qid_index[0] == 1:
            loop_number.append(int(split_column_name[0]))
        else:
            print('Issues with ' + question)
            break

        # Extract the numeric part of the QID (e.g., 'QID1' becomes ['QID', '1'])
        qid_numeric_parts = re.split('QID', split_column_name[qid_index[0]])
        question_number.append(int(qid_numeric_parts[1]))

        # Handle sub-questions
        if max_index > qid_index[0]:
            if split_column_name[qid_index[0] + 1].isnumeric():
                sub_question_number.append(int(split_column_name[qid_index[0] + 1]))
            else:
                sub_question_number.append(0)
        else:
            sub_question_number.append(0)
    
    # Create a DataFrame to sort the questions
    question_id_dict = {
        'question_column': question_columns,
        'question_number': question_number,
        'sub_question_number': sub_question_number,
        'loop_number': loop_number
    }
    question_id_df = pd.DataFrame(question_id_dict)
    question_id_df = question_id_df.sort_values(by=['question_number', 'sub_question_number', 'loop_number'], ascending=[True, True, True])

    # Extract the question columns in the new order
    question_df_sorted = responses_df.loc[:, question_id_df['question_column']]
    
    # Concatenate non-question and question columns
    final_results_df = pd.concat([non_question_df, question_df_sorted], axis=1)
    
    return final_results_df


# ========================================================
# Survey Block and Survey Flow Handling Functions
# ========================================================
def get_block_data(base_url, survey_id, token):
    """
    Fetches survey data from the Qualtrics API, extracts block names and their associated
    questions, and returns a DataFrame with the ordered blocks and question IDs.

    Args:
        base_url (str): The base URL for the Qualtrics API.
        survey_id (str): The unique ID of the survey to fetch.
        token (str): The API token for authentication.

    Returns:
        blocks_df (pd.DataFrame): A DataFrame with Block Name and Question ID columns.
    """
    # Make the API request to get the survey data
    survey_url = f'{base_url}/API/v3/surveys/{survey_id}'
    response = requests.get(survey_url, headers={"Authorization": f"Bearer {token}"})
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch survey data: {response.status_code}, {response.text}")
    
    # Parse the JSON response
    survey_question_dictionary = response.json()

    # Extract necessary survey details
    survey_result = survey_question_dictionary.get('result', {})
    blocks = survey_result.get('blocks', {})
    flow = survey_result.get('flow', [])

    # Process the survey flow to get ordered blocks
    ordered_blocks = process_survey_flow(flow, blocks)

    # Convert to a DataFrame
    blocks_df = pd.DataFrame(ordered_blocks)

    return blocks_df


def extract_block_details(block_id, blocks):
    """
    Extract details for a single block given its ID.
    Returns a list of dictionaries containing block name and question IDs.
    """
    block_details = blocks.get(block_id, {})
    block_name = block_details.get('description', 'No Description')
    elements = block_details.get('elements', [])
    question_ids = [element['questionId'] for element in elements if element['type'] == 'Question']

    return [{'Block Name': block_name, 'Question ID': question_id} for question_id in question_ids]


def process_survey_flow(flow_items, blocks):
    """
    Process the flow structure of the survey, handling both blocks and branches.
    Returns a list of dictionaries with block name and question IDs in the order they appear.
    """
    ordered_blocks = []
    for flow_item in flow_items:
        if flow_item.get('type') == 'Block':  # Standard block
            block_id = flow_item.get('id')
            ordered_blocks.extend(extract_block_details(block_id, blocks))
        elif flow_item.get('type') == 'Branch':  # Nested branch
            nested_flow = flow_item.get('flow', [])
            # Recursively process nested flow
            ordered_blocks.extend(process_survey_flow(nested_flow, blocks))  
    return ordered_blocks


# ========================================================
# Survey Questions and Question Value Handling Functions
# ========================================================
def get_survey_questions(base_url, token, survey_id):
    """
    Retrieve the questions from a specific survey in the Qualtrics API and clean the data by stripping HTML tags.

    This function fetches the survey questions from a specified survey using the Qualtrics API. 
    It returns the survey data as a dictionary and removes any HTML tags that may be present 
    in the survey question text.

    Args:
        base_url (str): The base URL for the Qualtrics API.
    token (str): The API token used for authorization.
        survey_id (str): The unique ID of the survey whose questions are being retrieved.

    Returns:
        dict: A dictionary containing the survey questions with HTML tags removed.
        
    Raises:
        requests.exceptions.RequestException: If there's an error with the API request.
    """
    # Set the URL for the specific survey
    survey_url = '{0}/API/v3/surveys/{1}'.format(base_url, survey_id)
    
    # Pull the survey data
    response = requests.get(survey_url, headers={"Authorization": "Bearer " + token})
    
    # Convert the data into a more readable format
    surveyq_dict = response.json()

    # Apply HTML cleaning on the survey questions inside 'result'
    if 'result' in surveyq_dict and 'questions' in surveyq_dict['result']:
        for question_id, question_data in surveyq_dict['result']['questions'].items():
            if 'questionText' in question_data:
                question_data['questionText'] = strip_html(question_data['questionText']).strip()
            if 'choices' in question_data:
                question_data['choices'] = strip_html(question_data['choices'])

    return surveyq_dict


def strip_html(data):
    """
    Remove HTML tags from the input data, which may be a string, list, or dictionary.
    
    Args:
        data (str, list, or dict): The data from which HTML tags should be stripped.
    
    Returns:
        The same data structure with HTML tags removed from strings.
    """
    html_pattern = re.compile(r'<[^>]*>')
    if isinstance(data, str):
        return html_pattern.sub('', data)
    elif isinstance(data, dict):
        return {k: strip_html(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [strip_html(item) for item in data]
    else:
        return data


def extract_column_data_types(question_dictionary, responses_df, base_url, token, survey_id):
    """
    Extracts column data types and question details from a survey.
    
    Parameters:
    - question_dictionary (dict): Survey dictionary with questions.
    - responses_df (pd.DataFrame): Survey response DataFrame.
    - base_url (str): Base URL for Qualtrics API.
    - token (str): API token.
    - survey_id (str): Survey ID.
    
    Returns:
    - question_df (pd.DataFrame): DataFrame containing question details.
    - question_values_df (pd.DataFrame): DataFrame containing question values and answer IDs.
    """
    
    # Extract column names from responses DataFrame
    column_names = responses_df.columns
    question_columns = [col for col in column_names if 'QID' in col]
    
    if len(question_columns) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Initialize lists for different question properties
    question_id_list = []
    question_name_list = []
    question_text_list = []
    question_type_list = []
    question_selector_list = []
    
    # Initialize lists for question details
    long_text_id_list = []
    question_value_list = []
    answer_id_list = []
    is_numeric_list = []
    keep_question_list = []
    
    # Initialize lists for group-related data
    group_question_id_list = []
    group_value_list = []
    group_answer_id_list = []
    
    # List of survey question keys
    key_list = list(question_dictionary.keys())

    for col in question_columns:
        split_column_name = re.split('_|#', col)  # Split on underscores or # for QID
        qid_index = [i for i in range(len(split_column_name)) if 'QID' in split_column_name[i]]

        if split_column_name[qid_index[0]] in key_list:
            current_question = question_dictionary.get(split_column_name[qid_index[0]])
            
            # Append basic question properties
            question_id_list.append(col)
            question_name_list.append(current_question.get('questionName'))
            current_type = current_question.get('questionType').get('type')
            ## new line below
            current_selector = current_question.get('questionType').get('selector')
            question_type_list.append(current_type)
            question_selector_list.append(current_question.get('questionType').get('selector'))

            # Check if question type is numeric
            is_numeric_list.append(is_numeric(current_question))
            
            # Handle different question types
            # first one is new
            if current_type == 'TE' and current_selector == 'FORM':
                handle_te_form_question(current_question, split_column_name, question_text_list, keep_question_list)
            elif current_type == 'Matrix':
                handle_matrix_question(current_question, split_column_name, question_text_list, long_text_id_list, question_value_list, answer_id_list, keep_question_list)
            elif current_type == 'CS':
                handle_cs_question(current_question, split_column_name, question_text_list, is_numeric_list, keep_question_list)
            elif current_type == 'RO':
                handle_ro_question(current_question, split_column_name, question_text_list, long_text_id_list, question_value_list, answer_id_list, keep_question_list, base_url, token, survey_id)
            elif current_type == 'Slider':
                handle_slider_question(current_question, split_column_name, question_text_list, is_numeric_list, keep_question_list, long_text_id_list, question_value_list, answer_id_list)
            elif current_type == 'Timing':
                handle_timing_question(current_question, question_text_list, is_numeric_list, keep_question_list)
            elif current_type == 'SS':
                handle_graphic_slider(current_question, question_selector_list, question_text_list, is_numeric_list, keep_question_list)
            elif current_type == 'PGR':
                handle_pgr_question(current_question, split_column_name, question_text_list, group_question_id_list, group_answer_id_list, group_value_list, keep_question_list)
            else:
                handle_default_question(current_question, split_column_name, question_text_list, long_text_id_list, question_value_list, answer_id_list, question_type_list, question_selector_list, keep_question_list)
    
    # Convert lists to numpy arrays and create DataFrames
    question_df, question_values_df = create_question_dataframes(
        question_id_list, 
        question_name_list, 
        question_text_list, 
        question_type_list, 
        question_selector_list, 
        is_numeric_list, 
        long_text_id_list, 
        question_value_list, 
        answer_id_list, 
        keep_question_list
    )
    
    return question_df, question_values_df


def is_numeric(current_question):
    """
    Checks if a question should be treated as numeric based on type or validation settings.
    """
    question_type = current_question.get('questionType', {}).get('type')
    
    # Treat Rank Order and Slider questions as numeric by default
    if question_type in ['RO', 'Slider']:
        return True
    
    # Check for validation settings indicating numeric input
    if 'validation' in current_question:
        current_validation = current_question.get('validation')
        if 'type' in current_validation and current_validation.get('type') == 'ValidNumber':
            return True
    
    return False

######## new function
def handle_te_form_question(current_question, split_column_name, 
                            question_text_list, keep_question_list):
    """Handle TE questions with a FORM selector."""
    main_question_text = current_question.get('questionText')  # e.g. 'Athlete Information'
    choices = current_question.get('choices', {})
    
    # The part after QID88_ is the choice key (1, 2, 3, etc.)
    choice_key = split_column_name[1]

    # Pull the sub-field text, e.g. "First Name:", "Last Name:"
    sub_field_text = choices.get(choice_key, {}).get('choiceText', '')
    
    # Combine main text and sub-field text
    question_text_list.append(f"{main_question_text}: {sub_field_text}")
    keep_question_list.append(True)
######

def handle_matrix_question(current_question, split_column_name, 
                           question_text_list, long_text_id_list, 
                           question_value_list, answer_id_list, keep_question_list):
    main_question_text = current_question.get('questionText')
    sub_question_text = current_question.get('subQuestions').get(split_column_name[1]).get('choiceText')
    question_text_list.append(f"{main_question_text}| {sub_question_text}")
    keep_question_list.append(True)
    
    # Ensure each sub-question ID (e.g., QID11_1) is correctly captured
    sub_question_id = split_column_name[0] + "_" + split_column_name[1]
    
    # Append answer choices for each sub-question with unique sub-question IDs
    choices = current_question.get('choices')
    for choice_key, choice in choices.items():
        long_text_id_list.append(sub_question_id)  # Use sub-question ID here
        answer_id_list.append(choice.get('recode'))
        # Use imageDescription if available, otherwise fallback to choiceText
        question_value_list.append(
            choice.get('imageDescription') or choice.get('choiceText')
        )


def handle_cs_question(current_question, split_column_name, question_text_list, is_numeric_list, keep_question_list):
    """
    Handles extraction for Cumulative Sum (CS) question types.
    """
    main_question_text = current_question.get('questionText')
    choices = current_question.get('choices')
    sub_question_text = choices.get(split_column_name[1]).get('choiceText')
    question_text_list.append(f"{main_question_text}| {sub_question_text}")
    is_numeric_list[-1] = True
    keep_question_list.append(True)


def handle_ro_question(current_question, split_column_name, question_text_list, long_text_id_list, question_value_list, answer_id_list, keep_question_list, base_url, token, survey_id):
    """
    Handles extraction for Rank Order (RO) question types, ensuring rank items are correctly added to question_values_df.
    """
    if split_column_name[-1] == 'TEXT':
        question_text_list.append(current_question.get('questionText'))
        keep_question_list.append(True)
    else:
        survey_info = get_full_survey_info(base_url, token, survey_id)
        question_id_prefix = split_column_name[0]  # Example: "QID10"
        
        # Retrieve choice order for the main question
        choice_order = survey_info['result']['Questions'][question_id_prefix].get('ChoiceOrder')
        
        # Get the rank position for this column (e.g., "1" in "QID10_1")
        rank_position = int(split_column_name[1]) - 1  # Adjust to zero-based indexing
        current_key = str(choice_order[rank_position])

        main_question_text = current_question.get('questionText')
        choices = current_question.get('choices')
        
        # Retrieve sub-question text for this rank option
        sub_question_text = choices.get(current_key).get('imageDescription') or choices.get(current_key).get('choiceText')
        question_text_list.append(f"{main_question_text} | {sub_question_text}")
        keep_question_list.append(True)

        # Append numeric ranks for each choice in the rank order question
        rank_question_id = f"{question_id_prefix}_{rank_position + 1}"
        for rank, choice_key in enumerate(choice_order, start=1):
            long_text_id_list.append(rank_question_id)  # Use unique rank item ID here
            answer_id_list.append(rank)
            question_value_list.append(str(rank))  # Rank values as strings (1, 2, 3, etc.)



def handle_slider_question(current_question, split_column_name, question_text_list, is_numeric_list, 
                           keep_question_list, long_text_id_list, question_value_list, answer_id_list):
    """
    Handles extraction for Slider question types, ensuring each possible value on the slider is added to question_values_df,
    with the full question_id including any suffix to distinguish sub-questions.
    
    IMPORTANT: qualtrics API doesnt return the number of stars available when pulling 'Choices'
    instead 'Choices' is how many slider sub-questions there are. Therefore, the slider_range below
    will need to be adjusted based on how many stars were available in the survey. 
    Suggest that we keep to max 5 to avoid issues with this.
    """
    main_question_text = current_question.get('questionText')
    choices = current_question.get('choices')
    
    if split_column_name[-1] == 'TEXT':
        question_text_list.append(main_question_text)
        keep_question_list.append(True)
    else:
        # Assuming sliders range from 1 to 5; adjust if range differs
        slider_range = range(1, 6)  # Replace with actual slider range if known
        full_question_id = f"{split_column_name[0]}_{split_column_name[1]}"  # e.g., "QID12_1"
        
        for value in slider_range:
            long_text_id_list.append(full_question_id)
            answer_id_list.append(value)
            question_value_list.append(str(value))  # Store slider values as strings
        
        sub_question_text = choices.get(split_column_name[1], {}).get('choiceText', '')
        question_text_list.append(f"{main_question_text} | {sub_question_text}")
        keep_question_list.append(True)


def handle_timing_question(current_question, question_text_list, is_numeric_list, keep_question_list):
    """
    Handles extraction for Timing question types.
    """
    question_text_list.append(current_question.get('questionText'))
    is_numeric_list[-1] = True
    keep_question_list.append(True)


def handle_graphic_slider(current_question, question_selector_list, question_text_list, is_numeric_list, keep_question_list):
    """
    Handles extraction for Graphic Slider (SS) question types.
    """
    if question_selector_list == 'TA':
        question_text_list.append(current_question.get('questionText'))
        is_numeric_list[-1] = True
        keep_question_list.append(True)
    else:
        print('Problems with Type SS (Graphical Slider)!!')


def handle_pgr_question(current_question, split_column_name, question_text_list, group_question_id_list, group_answer_id_list, group_value_list, keep_question_list):
    """
    Handles extraction for PGR (Pick, Group, Rank) question types.
    """
    if split_column_name[-1] == 'GROUP':
        main_question_text = current_question.get('questionText')
        groups = current_question.get('groups')
        sub_question_text = groups.get(split_column_name[1]).get('description')
        question_text_list.append(f"{main_question_text}| {sub_question_text}")
        keep_question_list.append(True)
        items = current_question.get('items')
        for item_key in items:
            current_item = items.get(item_key)
            group_question_id_list.append(split_column_name[0])
            group_answer_id_list.append(item_key)
            group_value_list.append(current_item.get('description'))


def handle_default_question(current_question, split_column_name, question_text_list, long_text_id_list, question_value_list, answer_id_list, question_type_list, question_selector_list, keep_question_list):
    """
    Handles the default case for question types not specifically handled.
    """
    question_text_list.append(current_question.get('questionText'))
    keep_question_list.append(True)
    if current_question.get('questionType').get('type') == 'MC':
        if split_column_name[-1] == 'TEXT':
            question_type_list[-1] = 'TE'
            question_selector_list[-1] = 'TE'
        else:
            choices = current_question.get('choices')
            for choice_key in choices:
                current_choice = choices.get(choice_key)
                long_text_id_list.append(split_column_name[0])
                answer_id_list.append(current_choice.get('recode'))
                # Use imageDescription if available, otherwise fallback to choiceText
                question_value_list.append(
                    current_choice.get('imageDescription') or current_choice.get('choiceText')
                )


def create_question_dataframes(question_id_list, question_name_list, question_text_list, question_type_list, question_selector_list, is_numeric_list, long_text_id_list, question_value_list, answer_id_list, keep_question_list):
    """
    Creates the final dataframes for question data and question values.
    """
    question_df = pd.DataFrame({
        "question_id": np.array(question_id_list)[np.array(keep_question_list)],
        "question_name": np.array(question_name_list)[np.array(keep_question_list)],
        "question_text": np.array(question_text_list)[np.array(keep_question_list)],
        "question_type": np.array(question_type_list)[np.array(keep_question_list)],
        "question_selector": np.array(question_selector_list)[np.array(keep_question_list)],
        "is_numeric": np.array(is_numeric_list)[np.array(keep_question_list)]
    })
    
    question_values_df = pd.DataFrame({
        "question_id": long_text_id_list,
        "question_value": question_value_list,
        "answer_id": answer_id_list
    })
    
    return question_df, question_values_df


def create_data_type_dictionary(question_df, question_values_df):
    """
    Creates a dictionary of data types for the questions and adds the corresponding data types to the question DataFrame.
    
    Args:
        question_df (pd.DataFrame): DataFrame containing the questions' metadata.
        question_values_df (pd.DataFrame): DataFrame containing the question values and answer IDs.
    
    Returns:
        pd.DataFrame: The updated question DataFrame with an added 'DataType' column.
    """
    # Get lists of the Free Text columns
    mask = np.array(question_df['question_type'] == 'TE') & np.array(question_df['is_numeric'] == False)
    free_text_columns = set(question_df['question_id'][mask])
    
    # Get lists of the FileUpload columns
    file_upload_columns = set(question_df['question_id'][question_df['question_type'] == 'FileUpload'])
    
    # Get list of Meta columns (categorical but without predefined categories)
    meta_columns = set(question_df['question_id'][question_df['question_type'] == 'Meta'])
    
    # Get lists of the Draw columns (often signatures)
    draw_columns = set(question_df['question_id'][question_df['question_type'] == 'Draw'])
    
    # Frequency Plots (Multiple Choice questions)
    multiple_choice_columns = set(question_values_df['question_id'])
    
    # Get Timing columns (tracks page times)
    timing_columns = set(question_df['question_id'][question_df['question_type'] == 'Timing'])
    
    # Get Date columns
    date_columns = set(question_df['question_id'][question_df['question_type'] == 'SBS'])
    
    # Get Rank Order columns
    rank_order_columns = set(question_df['question_id'][question_df['question_type'] == 'RO'])
    
    # Get Group columns for questions and responses
    group_columns = set(question_df['question_id'][question_df['question_type'] == 'PGR'])
    
    # Get the numeric columns
    # numeric_columns = set(question_df['question_id'][question_df['is_numeric']])
    numeric_columns = set(question_df['question_id'][question_df['is_numeric'] | question_df['question_id'].isin(rank_order_columns)])
    
    # Create a dictionary of all the different column types
    column_data_types = {
        'MultipleChoice': list(multiple_choice_columns),
        'Numeric': list(numeric_columns),
        'FreeText': list(free_text_columns),
        'RankOrder': list(rank_order_columns),
        'FileUpload': list(file_upload_columns),
        'Group': list(group_columns),
        'MetaData': list(meta_columns),
        'Draw': list(draw_columns),
        'Timing': list(timing_columns),
        'Dates': list(date_columns)
    }
    
    # Determine the data type for each question
    data_type = []
    for question_id, question_type, question_selector in zip(
        question_df['question_id'], question_df['question_type'], question_df['question_selector']):
        
        # Prioritize Rank Order
        if question_id in column_data_types.get('RankOrder', []):
            data_type.append('Numeric')
        elif question_id in column_data_types.get('Numeric', []):
            data_type.append('Numeric')
        elif question_id in column_data_types.get('MultipleChoice', []):
            data_type.append('MultipleChoice')
        elif question_id in column_data_types.get('FreeText', []):
            data_type.append('FreeText')
        elif question_id in column_data_types.get('FileUpload', []):
            data_type.append('FileUpload')
        elif question_id in column_data_types.get('Group', []):
            data_type.append('Group')
        elif question_id in column_data_types.get('MetaData', []):
            data_type.append('MetaData')
        elif question_id in column_data_types.get('Draw', []):
            data_type.append('Draw')
        elif question_id in column_data_types.get('Timing', []):
            data_type.append('Timing')
        elif question_id in column_data_types.get('Dates', []):
            data_type.append('Dates')
        elif question_type == 'Matrix' and question_selector == 'Likert':
            data_type.append('MultipleChoice')  # Assign 'MatrixLikert' type for Matrix-Likert questions
        else:
            data_type.append('Unknown')  # Default data type

    # Add the determined data types to the question DataFrame
    question_df['data_type'] = data_type
    return question_df


def reorder_question_df_with_normalized_ids(question_df, blocks_df):
    """
    Reorders the question_df DataFrame to match the Question ID order in blocks_df,
    creating a separate column for normalized Question IDs.

    Args:
        question_df (pd.DataFrame): DataFrame containing question metadata.
        blocks_df (pd.DataFrame): DataFrame with ordered Block Names and Question IDs.

    Returns:
        pd.DataFrame: A reordered question_df DataFrame.
    """
    # Add a separate column for normalized IDs
    question_df['normalized_id'] = question_df['question_id'].str.extract(r'^(QID\d+)')
    
    # Merge blocks_df with question_df based on the normalized ID
    merged_df = blocks_df.merge(question_df, how='left', left_on='Question ID', right_on='normalized_id')
    
    # Drop unnecessary columns and reorder
    reordered_question_df = merged_df.drop(columns=['normalized_id']).reset_index(drop=True)

    return reordered_question_df


def reorder_question_df(question_df, blocks_df):
    """
    Reorders the question_df DataFrame to match the Question ID order in blocks_df.

    Args:
        question_df (pd.DataFrame): DataFrame containing question metadata.
        blocks_df (pd.DataFrame): DataFrame with ordered Block Names and Question IDs.

    Returns:
        pd.DataFrame: A reordered question_df DataFrame.
    """
    # Filter question_df to include only Question IDs in blocks_df
    reordered_question_df = question_df.set_index('question_id').loc[blocks_df['Question ID']].reset_index()

    return reordered_question_df

# ========================================================
# Survey Response Cleaning Handling Functions
# ========================================================
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


def subset_by_date_range(responses_df, start_date, end_date):
    """
    Subset the responses DataFrame based on a specified date range.

    This function filters the `responses_df` DataFrame to include only rows where the 
    'recordedDate' column falls within the specified `start_date` and `end_date` range.
    The 'recordedDate' column must be in a timezone-aware datetime format (UTC). The 
    function will automatically convert `start_date` and `end_date` to UTC if they are 
    timezone-naive.

    Parameters:
    ----------
    responses_df : pd.DataFrame
        The DataFrame containing survey responses with a 'recordedDate' column in ISO format.
    start_date : str or datetime-like
        The start of the date range, inclusive. It should be in a format compatible with 
        `pd.to_datetime`.
    end_date : str or datetime-like
        The end of the date range, inclusive. It should be in a format compatible with 
        `pd.to_datetime`.

    Returns:
    -------
    pd.DataFrame
        A subset of `responses_df` where the 'recordedDate' is within the specified date range.
        
    Example:
    -------
    >>> responses_df = subset_by_date_range(responses_df, '2024-06-27', '2024-07-08')
    """

    # Ensure the 'recordedDate' column is in datetime format and set to UTC if needed
    responses_df['recordedDate'] = pd.to_datetime(responses_df['recordedDate']).dt.tz_convert('UTC')

    # Convert the start and end dates to datetime with UTC timezone
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Subset the DataFrame based on the date range
    subset_df = responses_df[(responses_df['recordedDate'] >= start_date) & (responses_df['recordedDate'] <= end_date)]
    
    return subset_df


# ========================================================
# Qualtrics Account Management Functions
# Users and Groups Handling Functions
# ========================================================
def get_users(base_url, token):
    """
    Fetches the list of users from the API.

    Args:
        base_url (str): The base URL for the API.
        token (str): The authorization token for the API.

    Returns:
        dict: A JSON object containing the list of users.
    """
    # Set the endpoint URL
    endpoint_url = '{0}/API/v3/users'.format(base_url)
    
    # Pull the survey data
    response = requests.get(endpoint_url, 
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token}) 
    
    # Convert the data into a more readable format
    response = response.json()    
    return response


def get_user_identity(base_url, token):
    """
    Retrieves the identity of the current user.

    Args:
        base_url (str): Base URL for the API.
        access_token (str): Access token for authentication.

    Returns:
        dict: User identity information.
    """
    identity_url = '{0}/API/v3/whoami'.format(base_url)
    response = requests.get(identity_url,
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token})
    return response.json()


def get_groups(base_url, token):
    """
    Retrieves all groups.

    Args:
        base_url (str): Base URL for the API.
        access_token (str): Access token for authentication.

    Returns:
        dict: Group information.
    """
    groups_url = '{0}/API/v3/groups'.format(base_url)
    response = requests.get(groups_url,
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token})
    return response.json()


def get_group_details(base_url, token, group_id):
    """
    Retrieve details for a single group by ID.

    Args:
        base_url (str): The base URL of your Qualtrics data center.
        token (str): OAuth2 bearer token used to authenticate.
        group_id (str): The ID of the group (e.g., 'GR_dms1ySSxMJNZkF0').

    Returns:
        dict: A JSON response containing details of the group, including 'type'.
    """
    url = f"{base_url}/API/v3/groups/{group_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    return response.json()


def add_user_to_group(base_url, token, group_id, user_id):
    """
    Adds a single user to a Qualtrics group.

    Args:
        base_url (str): Qualtrics base URL, e.g., https://{data_center}.qualtrics.com
        token (str): OAuth2 bearer token with write:groups scope.
        group_id (str): The Qualtrics Group ID, e.g., 'GR_9TttXzNhREpoOBE'
        user_id (str): The single user ID to add, e.g., 'UR_9HxQQrko6McPT82'

    Returns:
        dict: JSON response from Qualtrics API, typically:
              {
                  "meta": {
                      "httpStatus": "200 - OK",
                      "requestId": "...",
                      "notice": ""
                  }
              }
    """
    endpoint_url = f"{base_url}/API/v3/groups/{group_id}/members"
    
    # The body must have just "userId" per Qualtrics docs:
    data = {
        "userId": user_id
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.post(endpoint_url, headers=headers, json=data)
    return response.json()


def list_users_in_group(base_url, token, group_id, offset=0):
    """
    Retrieves a list of users in a specific Qualtrics group.

    Args:
        base_url (str): The base URL of your Qualtrics data center (e.g., 'https://{data_center}.qualtrics.com').
        token (str): OAuth2 bearer token with read:groups scope.
        group_id (str): The Qualtrics Group ID, e.g., 'GR_12345abcdef'.
        offset (int): Pagination offset, default is 0.

    Returns:
        dict: JSON response from the Qualtrics API, including:
              - result['elements']: List of users in the group.
              - result['nextPage']: URL for the next page of results (if applicable).
    """
    endpoint_url = f"{base_url}/API/v3/groups/{group_id}/members"
    params = {
        "offset": offset
    }
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(endpoint_url, headers=headers, params=params)
    return response.json()

# ========================================================
# Directory Handling Functions
# ========================================================
def list_directories(base_url, token):
    """
    List all XM Directories available to the authenticated user.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token.

    Returns:
        dict: JSON response containing a list of directories and their IDs.
    """
    endpoint_url = f"{base_url}/API/v3/directories"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers)
    return response.json()


# ========================================================
# Library Handling Functions
# ========================================================
def list_libraries(base_url, token):
    """
    Fetch all libraries available to the caller in Qualtrics.

    Parameters:
        base_url (str): The base URL for the Qualtrics API.
        token (str): Bearer token for authorization.

    Returns:
        dict: Parsed JSON response from the Qualtrics API containing the list of libraries.

    Raises:
        Exception: If the API request fails.
    """
    endpoint_url = f"{base_url}/API/v3/libraries"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(endpoint_url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching libraries: {e}")



def list_library_messages(base_url, token, library_id, category=None, offset=0):
    """
    Fetch all messages from a specified library in Qualtrics.

    Parameters:
        base_url (str): The base URL for the Qualtrics API.
        token (str): Bearer token for authorization.
        library_id (str): The ID of the library to fetch messages from.
        category (str, optional): The category a message belongs to (e.g., invite, thankYou).
        offset (int, optional): The starting position for pagination. Default is 0.

    Returns:
        dict: Parsed JSON response from the Qualtrics API containing the list of messages.

    Raises:
        Exception: If the API request fails.
    """
    endpoint_url = f"{base_url}/API/v3/libraries/{library_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    params = {
        "category": category,
        "offset": offset
    }

    # Remove None values from params to avoid sending unnecessary keys
    params = {key: value for key, value in params.items() if value is not None}

    try:
        response = requests.get(endpoint_url, headers=headers, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching library messages: {e}")



def create_library_message(base_url, token, library_id, description, category, messages):
    """
    Creates a new message in a specified Qualtrics library.

    Args:
        base_url (str): The base URL of your Qualtrics data center, 
                        e.g., 'https://{data_center}.qualtrics.com'.
        token (str): OAuth2 bearer token with the 'write:libraries' scope.
        library_id (str): The Qualtrics Library ID (e.g., 'UR_12345abcdef').
        description (str): Description for the new message.
        category (str): The category for the new message. Possible values include:
                        'invite', 'thankYou', 'reminder', 'endOfSurvey', 
                        'inactiveSurvey', 'general', 'lookAndFeel', 'emailSubject',
                        'smsInvite', 'smsReminder', 'smsThankYou', 'validation',
                        'evaluatorInvite', 'evaluatorReminder', 'subjectLine'.
        messages (dict): A dictionary of language-code keys and message text values.
                         Example: {"en": "Example English message.", "pt-br": "Mensagem em Português."}

    Returns:
        dict: JSON response from the Qualtrics API. Example structure:
              {
                  "result": {
                      "id": "string"  # The newly created message ID
                  },
                  "meta": {
                      "httpStatus": "string",
                      "requestId": "string",
                      "notice": "string"
                  }
              }
    """
    # Construct the endpoint URL
    endpoint_url = f"{base_url}/API/v3/libraries/{library_id}/messages"
    
    # Prepare the request payload
    payload = {
        "description": description,
        "category": category,
        "messages": messages
    }

    # Set the request headers, including the OAuth2 Bearer token
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Execute the POST request
    response = requests.post(endpoint_url, headers=headers, json=payload)
    
    # Return the JSON response
    return response.json()


def get_library_message(base_url, token, library_id, message_id):
    """
    Retrieves a specific message from a Qualtrics library.

    Args:
        base_url (str): The base URL of your Qualtrics data center,
                        e.g., 'https://{data_center}.qualtrics.com'.
        token (str): OAuth2 bearer token with the 'read:libraries' scope.
        library_id (str): The Qualtrics Library ID (e.g., 'UR_12345abcdef').
        message_id (str): The Qualtrics Message ID (e.g., 'MS_67890abcxyz').

    Returns:
        dict: JSON response from the Qualtrics API. Example structure:
              {
                  "result": {
                      "category": "string",
                      "description": "string",
                      "messages": {
                        "property1": "string",
                        "property2": "string"
                      }
                  },
                  "meta": {
                      "httpStatus": "string",
                      "requestId": "string",
                      "notice": "string"
                  }
              }
    """
    # Construct the endpoint URL
    endpoint_url = f"{base_url}/API/v3/libraries/{library_id}/messages/{message_id}"
    
    # Prepare the request headers, including the OAuth2 Bearer token
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Execute the GET request
    response = requests.get(endpoint_url, headers=headers)
    response.raise_for_status()  # (Optional) Raise HTTPError for bad responses
    return response.json()

# ========================================================
# Mailing List Handling Functions
# ========================================================
def list_mailing_lists(base_url, token, directory_id, page_size=100, skip_token=None, include_count=True):
    """
    Retrieves a list of mailing lists in an XM Directory.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token with read:mailing_lists scope.
        directory_id (str): Directory ID (POOL_...).
        page_size (int): Maximum number of items to return per request (default 100).
        skip_token (str): Token for pagination (default None).
        include_count (bool): Include contact count (default False).

    Returns:
        dict: JSON response containing mailing list information.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists"
    params = {
        "pageSize": page_size,
        "includeCount": include_count
    }
    if skip_token:
        params["skipToken"] = skip_token

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers, params=params)
    return response.json()


def update_mailing_list_owner(base_url, token, directory_id, mailing_list_id, owner_id):
    """
    Updates the owner of a mailing list to share it with a group.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `write:mailing_lists` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        owner_id (str): Group ID to set as the owner.

    Returns:
        dict: JSON response from the API.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}"
    data = {
        "ownerId": owner_id
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.put(endpoint_url, headers=headers, json=data)
    return response.json()


def create_mailing_list(base_url, token, directory_id, name, owner_id=None, prioritize_metadata=False):
    """
    Creates a new mailing list in an XM Directory.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token with write:mailing_lists scope.
        directory_id (str): Directory ID (POOL_...).
        name (str): Name of the mailing list.
        owner_id (str): Owner of the mailing list (default None).
        prioritize_metadata (bool): Prioritize list metadata over contact metadata (default False).

    Returns:
        dict: JSON response with the newly created mailing list's ID.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists"
    data = {
        "name": name,
        "prioritizeListMetadata": prioritize_metadata
    }
    if owner_id:
        data["ownerId"] = owner_id

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint_url, headers=headers, json=data)
    return response.json()


def get_mailing_list(base_url, token, directory_id, mailing_list_id, include_count=True):
    """
    Retrieves details of a specific mailing list.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token with read:mailing_lists scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing list ID (CG_...).
        include_count (bool): Include contact count (default False).

    Returns:
        dict: JSON response containing mailing list details.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}"
    params = {
        "includeCount": include_count
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers, params=params)
    return response.json()


def update_mailing_list(base_url, token, directory_id, mailing_list_id, name, owner_id=None):
    """
    Updates a mailing list's name or owner.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token with write:mailing_lists scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing list ID (CG_...).
        name (str): New name for the mailing list.
        owner_id (str): New owner of the mailing list (default None).

    Returns:
        dict: JSON response from the API.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}"
    data = {
        "name": name
    }
    if owner_id:
        data["ownerId"] = owner_id

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.put(endpoint_url, headers=headers, json=data)
    return response.json()


def delete_mailing_list(base_url, token, directory_id, mailing_list_id):
    """
    Deletes a mailing list from an XM Directory.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token with write:mailing_lists scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing list ID (CG_...).

    Returns:
        dict: JSON response from the API.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.delete(endpoint_url, headers=headers)
    return response.json()


# ========================================================
# Contact List WITHIN Mailing List Handling Functions
# ========================================================
def list_contacts_in_mailing_list(base_url, token, directory_id, mailing_list_id, page_size=50, skip_token=None, include_embedded=False):
    """
    Retrieves a list of contacts in a specified mailing list.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `read:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        page_size (int): Number of contacts to retrieve per page (default: 50).
        skip_token (str): Token for pagination (default: None).
        include_embedded (bool): Include embedded data in the response.

    Returns:
        dict: JSON response containing contacts and pagination info.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/contacts"
    params = {
        "pageSize": page_size,
        "includeEmbedded": 'true' if include_embedded else 'false'
    }
    if skip_token:
        params["skipToken"] = skip_token

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers, params=params)
    return response.json()


def create_contact_in_mailing_list(base_url, token, directory_id, mailing_list_id, first_name, last_name, email, phone=None, ext_ref=None, embedded_data=None, private_embedded_data=None, language=None, unsubscribed=False):
    """
    Creates a contact in a specified mailing list.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `write:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        first_name (str): Contact's first name.
        last_name (str): Contact's last name.
        email (str): Contact's email.
        phone (str): Contact's phone number (optional).
        ext_ref (str): External reference for the contact (optional).
        embedded_data (dict): Additional metadata for the contact (optional).
        private_embedded_data (dict): Private metadata for the contact (optional).
        language (str): Language code for the contact (optional).
        unsubscribed (bool): Contact's subscription status (default: False).

    Returns:
        dict: JSON response containing the created contact's ID.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/contacts"
    data = {
        "firstName": first_name,
        "lastName": last_name,
        "email": email,
        "phone": phone,
        "extRef": ext_ref,
        "embeddedData": embedded_data,
        "privateEmbeddedData": private_embedded_data,
        "language": language,
        "unsubscribed": unsubscribed
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint_url, headers=headers, json={k: v for k, v in data.items() if v is not None})
    return response.json()


def list_bounced_mailing_list_contacts(base_url, token, directory_id, mailing_list_id, page_size=50, skip_token=None, since=None):
    """
    Retrieves a list of bounced contacts in a specified mailing list.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `read:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        page_size (int): Number of contacts to retrieve per page (default: 50).
        skip_token (str): Token for pagination (default: None).
        since (str): Date-time string to filter bounced contacts (optional).

    Returns:
        dict: JSON response containing bounced contacts.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/bouncedContacts"
    params = {
        "pageSize": page_size,
        "skipToken": skip_token,
        "since": since
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers, params={k: v for k, v in params.items() if v is not None})
    return response.json()


def list_opted_out_mailing_list_contacts(base_url, token, directory_id, mailing_list_id, page_size=50, skip_token=None, since=None):
    """
    Retrieves a list of opted-out contacts in a specified mailing list.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `read:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        page_size (int): Number of contacts to retrieve per page (default: 50).
        skip_token (str): Token for pagination (default: None).
        since (str): Date-time string to filter opted-out contacts (optional).

    Returns:
        dict: JSON response containing opted-out contacts.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/optedOutContacts"
    params = {
        "pageSize": page_size,
        "skipToken": skip_token,
        "since": since
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers, params={k: v for k, v in params.items() if v is not None})
    return response.json()


def get_mailing_list_contact(base_url, token, directory_id, mailing_list_id, contact_id):
    """
    Retrieves a specific contact in a mailing list.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `read:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        contact_id (str): Contact ID (CID_...).

    Returns:
        dict: JSON response containing contact details.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/contacts/{contact_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(endpoint_url, headers=headers)
    return response.json()


def update_mailing_list_contact(base_url, token, directory_id, mailing_list_id, contact_id, **kwargs):
    """
    Updates a contact in a mailing list.

    Args:
        base_url (str): Qualtrics base URL.
        token (str): OAuth2 bearer token with `write:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        contact_id (str): Contact ID (CID_...).
        kwargs: Attributes to update (e.g., firstName, lastName, email, etc.).

    Returns:
        dict: JSON response containing the update status.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/contacts/{contact_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.put(endpoint_url, headers=headers, json={k: v for k, v in kwargs.items() if v is not None})
    return response.json()


def delete_contact_from_mailing_list(base_url, token, directory_id, mailing_list_id, contact_id):
    """
    Deletes a single contact from a specified mailing list.

    Args:
        base_url (str): Qualtrics base URL (e.g., https://{data_center}.qualtrics.com).
        token (str): OAuth2 bearer token with `write:mailing_list_contacts` scope.
        directory_id (str): Directory ID (POOL_...).
        mailing_list_id (str): Mailing List ID (CG_...).
        contact_id (str): Contact ID (CID_...).

    Returns:
        dict: JSON response containing the status of the deletion.
    """
    endpoint_url = f"{base_url}/API/v3/directories/{directory_id}/mailinglists/{mailing_list_id}/contacts/{contact_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.delete(endpoint_url, headers=headers)
    return response.json()


# ========================================================
# Survey Editing Handling Functions
# ========================================================
def get_survey_meta(base_url, token, survey_id):
    """
    Fetches the metadata for a given survey from the API.

    Args:
        base_url (str): The base URL for the API.
        token (str): The authorization token for the API.
        survey_id (str): The ID of the survey to fetch metadata for.

    Returns:
        dict: A JSON object containing the survey's metadata.
    """
    # Set the endpoint URL
    endpoint_url = '{0}/API/v3/survey-definitions/{1}/metadata'.format(base_url, survey_id)
    
    # Pull the survey metadata
    response = requests.get(endpoint_url, 
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token}) 
    
    # Convert the data into a more readable format
    response = response.json()    
    return response


def change_survey_metadata(base_url, token, survey_id, metadata):
    """
    Updates a survey's metadata.

    Args:
        base_url (str): Base URL for the API.
        access_token (str): Access token for authentication.
        survey_id (str): Survey ID.
        metadata (dict): Metadata to update.

    Returns:
        dict: Response from the API.
    """
    survey_metadata_url = '{0}/API/v3/survey-definitions/{1}/metadata'.format(base_url, survey_id)
    response = requests.put(survey_metadata_url,
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token},
                            data=metadata)
    return response.json()


def update_survey(base_url, token, survey_id, data):
    """
    Updates a survey's details.

    Args:
        base_url (str): Base URL for the API.
        access_token (str): Access token for authentication.
        survey_id (str): Survey ID.
        data (dict): Data to update.

    Returns:
        dict: Response from the API.
    """
    survey_update_url = '{0}/API/v3/surveys/{1}'.format(base_url, survey_id)
    response = requests.put(survey_update_url,
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token},
                            data=data)
    return response.json()


def get_full_survey_info(base_url, token, survey_id):
    """
    Retrieves full survey information, pulling more detailed info.

    Args:
        base_url (str): Base URL for the API.
        access_token (str): Access token for authentication.
        survey_id (str): Survey ID.

    Returns:
        dict: Full survey information.
    """
    full_survey_info_url = '{0}/API/v3/survey-definitions/{1}'.format(base_url, survey_id)
    response = requests.get(full_survey_info_url,
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token})
    return response.json()


def share_survey(base_url, token, survey_id, data):
    """
    Shares a survey with a group or person.

    Args:
        base_url (str): Base URL for the API.
        access_token (str): Access token for authentication.
        survey_id (str): Survey ID.
        data (dict): Data to share the survey.

    Returns:
        dict: Response from the API.
    """
    share_survey_url = '{0}/API/v3/surveys/{1}/permissions/collaborations'.format(base_url, survey_id)
    response = requests.post(share_survey_url,
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + token},
                             data=data)
    return response.json()





# yul1.qualtrics.com/API/v3/surveys/{survey_id}/permissions/collaborations
# 

# curl --request POST \
#   --url https://yul1.qualtrics.com/API/v3/surveys/survey_id/permissions/collaborations \
#   --header 'Content-Type: application/json' \
#   --header 'X-API-TOKEN: ' \
#   --data '{
#   "recipientId": "string",
#   "userId": "string",
#   "permissions": {
#     "property1": {
#       "property1": true,
#       "property2": true
#     },
#     "property2": {
#       "property1": true,
#       "property2": true
#     }
#   }
# }'