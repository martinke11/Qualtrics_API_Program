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


# Build a function to build out the data argument
def def_data(**kwargs):
    # Convert to a string
    r = str(kwargs)
    return r


def def_data1(dic):
    # Some data formats in Qualtrics are different. So need to change the format
    # Input must be a dictionary.
    # First convert to a string
    r = str(dic)
    # Now change the quotes around
    r = r.replace("'", '"')
    return r


def def_data_t(**kwargs):
    # Convert to a dictionary
    r = kwargs
    return r


# API functions
# Get a token that will last an hour
def get_token(base_url, client_id, client_secret, data):
    tkn_url = base_url + "/oauth2/token"

    r = requests.post(tkn_url, auth=(client_id, client_secret), data=data)

    return r.json()


# Get a list of Users
def get_users(base_url, tkn):
    # Set the URL
    s = '{0}/API/v3/users'.format(base_url)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    r = r.json()
    return r


# Get a survey's metadata
def get_survey_meta(base_url, tkn, survey_id):
    # Set the URL
    s = '{0}/API/v3/survey-definitions/{1}/metadata'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    r = r.json()
    return r


# Change a survey's metadata
def put_survey_meta(base_url, tkn, survey_id, data):
    # Set the URL
    s = '{0}/API/v3/survey-definitions/{1}/metadata'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.put(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn},
                     data=data)
    # Convert the data into a more readable format
    r = r.json()
    return r


# Update a survey
def update_survey(base_url, tkn, survey_id, data):
    # Set the URL
    s = '{0}/API/v3/surveys/{1}'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.put(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn},
                     data=data)
    # Convert the data into a more readable format
    r = r.json()
    return r


# Pull the survey list
def get_surveys(base_url, tkn):
    # Extract all the surveys and their ids
    # Set Next Page to Data Type None Flag to True
    flag = True

    # Set a counter
    count = 0

    # Set cnum to blank
    cnum = ""

    while flag:
        # Pull the data
        if count == 0:
            # Set the URL
            s = '{0}/API/v3/surveys'.format(base_url)
        else:
            # Set the URL
            s = '{0}/API/v3/surveys?offset={1}'.format(base_url, cnum)

        # Pull the surveys
        r = requests.get(s, headers={"Authorization": "Bearer " + tkn})
        # Convert the data into a more readable format
        r = r.json()

        # Extract the Last of the Survey List
        cur_df = pd.DataFrame(r.get("result").get("elements"))

        if count == 0:
            # There is just one page of results
            survey_df = cur_df
        else:
            # Else add to the current dataframe
            survey_df = pd.concat([survey_df, cur_df])

        if r.get("result").get("nextPage") is None:
            # Set the flag to False to end the loop
            flag = False
        else:
            # Then the flag is still true
            # Extract the Next Page string
            np_str = r.get("result").get("nextPage")
            # Extract the current number for offset
            cnum = np_str.split("?offset=", 1)[1]
            # Update the counter
            count += 1

    # Reset the indices
    survey_df = survey_df.reset_index(drop=True)
    return survey_df


# Pull survey questions
def get_survey_qs(base_url, tkn, survey_id):
    # Set the URL
    s = '{0}/API/v3/surveys/{1}'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.get(s, headers={"Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    survey_q_dict = r.json()

    def strip_html(data):
        html_pattern = re.compile(r'<[^>]*>')
        if isinstance(data, str):
            return html_pattern.sub('', data)
        elif isinstance(data, dict):
            return {k: strip_html(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [strip_html(item) for item in data]
        else:
            return data

    # Apply HTML cleaning on the survey questions inside 'result'
    if 'result' in survey_q_dict and 'questions' in survey_q_dict['result']:
        for question_id, question_data in survey_q_dict['result']['questions'].items():
            if 'questionText' in question_data:
                question_data['questionText'] = strip_html(question_data['questionText'])
            if 'choices' in question_data:
                question_data['choices'] = strip_html(question_data['choices'])

    return survey_q_dict


# Pulls more information than the survey Qs
def get_full_survey(base_url, tkn, survey_id):
    # Set the URL
    s = '{0}/API/v3/survey-definitions/{1}'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    r = r.json()
    return r


# Post the survey Response Export
def get_survey_rx(base_url, tkn, survey_id):
    # Set the URL
    s = '{0}/API/v3/surveys/{1}/export-responses'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.post(s,
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + tkn},
                      data='{"format": "json", "compress": false}')
    # Convert the data into a more readable format
    r = r.json()
    return r


# Function to get the Response Export Progress
def get_survey_rxp(base_url, tkn, survey_id, ep_id):
    # Set the URL
    s = '{0}/API/v3/surveys/{1}/export-responses/{2}'.format(base_url, survey_id, ep_id)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    r = r.json()
    return r


# Get the survey responses
def get_survey(base_url, tkn, survey_id, fid):
    # Set the URL
    s = '{0}/API/v3/surveys/{1}/export-responses/{2}/file'.format(base_url, survey_id, fid)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    return r


def get_groups(base_url, tkn):
    # Set the URL
    s = '{0}/API/v3/groups'.format(base_url)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    r = r.json()
    return r


# Share a Survey with a group or person
def post_collabs(base_url, tkn, survey_id, data):
    # Set the URL
    s = '{0}/API/v3/surveys/{1}/permissions/collaborations'.format(base_url, survey_id)
    # Pull the survey data
    r = requests.post(s,
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + tkn},
                      data=data)
    # Convert the data into a more readable format
    r = r.json()
    return r


# Share a Survey with a group or person
def get_whoami(base_url, tkn):
    # Set the URL
    s = '{0}/API/v3/whoami'.format(base_url)
    # Pull the survey data
    r = requests.get(s,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    r = r.json()
    return r


recollab_json = open('RandE_Collaborator.json')


def organize_responses(responses):
    # DF = Dataframe of the responses
    # For each response extract and put in a DataFrame
    df = []
    for j in responses:
        # Extract the current response and normalize
        temp_df = pd.json_normalize(j.get('values'))
        if len(df) == 0:
            # Then temp_df = df
            df = temp_df
        else:
            # Then concatenate
            df = pd.concat([df, temp_df])

    # Reset the DataFrame index
    df = df.reset_index(drop=True)

    # Reorder the columns so that they appear in order
    # Find the QID columns
    col_names = df.columns
    q_cols = [x for x in col_names if x.startswith('QID')]
    # Non-QID columns
    non_q_cols = list(set(col_names) - set(q_cols))
    df_temp1 = df.loc[:, np.isin(col_names, non_q_cols)]

    # Create a function to order the column headers
    q_num = []
    q_subq = []
    q_loopn = []

    for q in q_cols:
        # Split the QID
        cur_spl = re.split('_|#', q)
        # Find the QID index
        qid_ind = [i for i in range(len(cur_spl)) if 'QID' in cur_spl[i]]
        # Get the max index
        max_ind = len(cur_spl) - 1

        # Determine where the QID is
        if qid_ind[0] == 0:
            # Then this is not a loop question
            q_loopn.append(0)
        elif qid_ind[0] == 1:
            # Then this is a loop question
            q_loopn.append(int(cur_spl[0]))
        else:
            print('Issues with ' + q)
            break

        # Grab the question
        q_spl = re.split('QID', cur_spl[qid_ind[0]])
        # Append the number
        q_num.append(int(q_spl[1]))

        # Determine if there is a subquestion
        if max_ind > qid_ind[0]:
            # Then there could be a subquestion... check
            if cur_spl[qid_ind[0] + 1].isnumeric():
                # Then get the subquestion
                q_subq.append(int(cur_spl[qid_ind[0] + 1]))
            else:
                # There is no subquestion
                q_subq.append(0)
        else:
            # There is no subquestion
            q_subq.append(0)

    # Create a DataFrame for the question columns
    tdic = {
        'QCol': q_cols,
        'Q_Num': q_num,
        'Q_SubQ': q_subq,
        'Q_LoopN': q_loopn
    }
    qdf = pd.DataFrame(tdic)
    qdf = qdf.sort_values(by=['Q_Num', 'Q_SubQ', 'Q_LoopN'], ascending=[True, True, True])

    # Get the second DataFrame
    df_temp2 = df.loc[:, qdf['QCol']]

    # Concatenate
    dfn = pd.concat([df_temp1, df_temp2], axis=1)

    return dfn


def clean_qs(questions_df):
    # Clean the questions DataFrame
    # Set the pattern that contains superfluous text formatting
    pat = r'(?=\<).+?(?<=\>)'
    for i, rows in questions_df.iterrows():
        # Grab the current text
        cur_t = questions_df.loc[i, 'QText']
        # Remove superfluous text
        new_t = re.sub(pat, "", cur_t)
        # Now do the same for any other patterns that don't translate well
        if '&#39;' in new_t:
            # This is an apostrophe
            pat1 = '&#39;'
            new_t = re.sub(pat1, "'", new_t)
        if '\n' in new_t:
            pat2 = '\n'
            new_t = re.sub(pat2, "", new_t)
        # Now fix the QText
        questions_df.loc[i, 'QText'] = new_t
    return questions_df


def extract_column_data_types(q_dic, rdf, base_url, tkn, survey_id):
    # Pull the column names
    col_names = rdf.columns
    # Find the columns that start with QID
    q_cols = [x for x in col_names if 'QID' in x]

    # Check to see if there are any questions
    if len(q_cols) > 0:

        # Extract the question text, type, value, and ID
        q_id = []
        q_name = []
        q_text = []
        q_type = []
        q_type_sel = []
        q_id_lt = []
        q_value = []
        q_ans_id = []
        q_numeric = []
        q_keep = []

        # Create a different set for groups
        qg_q_id = []
        qg_value = []
        qg_ans_id = []

        # Get list of survey question keys
        key_list = list(q_dic.keys())

        for x in q_cols:
            # Split the current QID
            cur_spl = re.split('_|#', x)
            # Find the QID index
            qid_ind = [i for i in range(len(cur_spl)) if 'QID' in cur_spl[i]]

            # Check to see if the key is present
            if cur_spl[qid_ind[0]] in key_list:
                # Record the QID column
                q_id.append(x)
                # Pull the question key
                cur_q = q_dic.get(cur_spl[qid_ind[0]])
                # Record the question name
                q_name.append(cur_q.get('questionName'))
                # Get the question type
                cur_type = cur_q.get('questionType').get('type')
                q_type.append(cur_type)
                cur_type_sel = cur_q.get('questionType').get('selector')
                q_type_sel.append(cur_type_sel)

                # Check to see if the type is numeric
                if 'validation' in cur_q:
                    cur_valid = cur_q.get('validation')
                    if 'type' in cur_valid:
                        cur_valid_type = cur_valid.get('type')
                        if 'ValidNumber' in cur_valid_type:
                            q_numeric.append(True)
                        else:
                            q_numeric.append(False)
                    else:
                        q_numeric.append(False)
                else:
                    q_numeric.append(False)

                if cur_type == 'Matrix':
                    qt1 = cur_q.get('questionText')
                    qt2 = cur_q.get('subQuestions').get(cur_spl[qid_ind[0] + 1]).get('choiceText')
                    qt = qt1 + '| ' + qt2
                    q_text.append(qt)
                    q_keep.append(True)

                    scores = cur_q.get('subQuestions').get(cur_spl[qid_ind[0] + 1]).get('scoring')
                    if cur_type_sel == 'CS':
                        q_numeric[-1] = True
                    else:
                        if scores is None:
                            choices = cur_q.get('choices')
                            for xx in choices:
                                cur_choice = choices.get(xx)
                                q_id_lt.append(x)
                                q_ans_id.append(cur_choice.get('recode'))
                                q_value.append(cur_choice.get('choiceText'))
                        else:
                            for xx in scores:
                                q_id_lt.append(x)
                                q_value.append(xx.get('value'))
                                if xx.get('answerId') is None:
                                    q_ans_id.append(xx.get('value'))
                                else:
                                    q_ans_id.append(xx.get('answerId'))
                elif cur_type == 'CS':
                    q_numeric[-1] = True
                    qt1 = cur_q.get('questionText')
                    choices = cur_q.get('choices')
                    qt2 = choices.get(cur_spl[qid_ind[0] + 1]).get('choiceText')
                    qt = qt1 + '| ' + qt2
                    q_text.append(qt)
                    q_keep.append(True)
                elif cur_type == 'RO':
                    if cur_spl[len(cur_spl) - 1] == 'TEXT':
                        q_type[-1] = 'TE'
                        q_type_sel[-1] = 'TE'
                        qt = cur_q.get('questionText')
                        q_text.append(qt)
                        q_keep.append(True)
                    else:
                        f_surv = get_full_survey(base_url, tkn, survey_id)
                        co = f_surv.get('result').get('Questions').get(cur_spl[qid_ind[0]]).get('ChoiceOrder')
                        cur_key = str(co[int(cur_spl[qid_ind[0] + 1]) - 1])
                        qt1 = cur_q.get('questionText')
                        choices = cur_q.get('choices')
                        qt2 = choices.get(cur_key).get('choiceText')
                        qt = qt1 + '| ' + qt2
                        q_text.append(qt)
                        q_keep.append(True)
                elif cur_type == 'Slider':
                    q_numeric[-1] = True
                    qt1 = cur_q.get('questionText')
                    choices = cur_q.get('choices')
                    qt2 = choices.get(cur_spl[qid_ind[0] + 1]).get('choiceText')
                    qt = qt1 + '| ' + qt2
                    q_text.append(qt)
                    q_keep.append(True)
                elif cur_type == 'Timing':
                    q_numeric[-1] = True
                    q_text.append(cur_q.get('questionText'))
                    q_keep.append(True)
                elif cur_type == 'SS':
                    if cur_type_sel == 'TA':
                        q_numeric[-1] = True
                        q_text.append(cur_q.get('questionText'))
                        q_keep.append(True)
                    else:
                        print('Problems with Type SS (Graphical Slider)!!')
                elif cur_type == 'PGR':
                    if cur_spl[len(cur_spl) - 1] == 'GROUP':
                        qt1 = cur_q.get('questionText')
                        groups = cur_q.get('groups')
                        qt2 = groups.get(cur_spl[qid_ind[0] + 1]).get('description')
                        qt = qt1 + '| ' + qt2
                        q_text.append(qt)
                        q_keep.append(True)
                        items = cur_q.get('items')
                        for xx in items:
                            cur_item = items.get(xx)
                            qg_q_id.append(x)
                            qg_ans_id.append(xx)
                            qg_value.append(cur_item.get('description'))
                    elif cur_spl[len(cur_spl) - 1] == 'RANK':
                        q_text.append(cur_q.get('questionText'))
                        q_keep.append(False)
                    elif cur_spl[len(cur_spl) - 1] == 'TEXT':
                        qt1 = cur_q.get('questionText')
                        items = cur_q.get('items')
                        cur_item = items.get(cur_spl[qid_ind[0] + 1])
                        qt2 = cur_item.get('description')
                        qt = qt1 + '| ' + qt2
                        q_text.append(qt)
                        q_keep.append(True)
                        q_type[-1] = 'TE'
                        q_type_sel[-1] = 'TE'
                    else:
                        q_text.append(cur_q.get('questionText'))
                        q_keep.append(True)
                        items = cur_q.get('items')
                        for xx in items:
                            cur_item = items.get(xx)
                            qg_q_id.append(x)
                            qg_ans_id.append(xx)
                            qg_value.append(cur_item.get('description'))
                else:
                    q_text.append(cur_q.get('questionText'))
                    q_keep.append(True)
                    if cur_q.get('questionType').get('type') == 'MC':
                        if cur_spl[-1] == 'TEXT':
                            q_type[-1] = 'TE'
                            q_type_sel[-1] = 'TE'
                        else:
                            choices = cur_q.get('choices')
                            for xx in choices:
                                cur_choice = choices.get(xx)
                                q_id_lt.append(x)
                                q_ans_id.append(cur_choice.get('recode'))
                                q_value.append(cur_choice.get('choiceText'))

        q_id = np.array(q_id)
        q_name = np.array(q_name)
        q_text = np.array(q_text)
        q_type = np.array(q_type)
        q_type_sel = np.array(q_type_sel)
        q_numeric = np.array(q_numeric)

        dic = {
            "QID": q_id[np.array(q_keep)],
            "QName": q_name[np.array(q_keep)],
            "QText": q_text[np.array(q_keep)],
            "QType": q_type[np.array(q_keep)],
            "QTypeSel": q_type_sel[np.array(q_keep)],
            "Q_Numeric": q_numeric[np.array(q_keep)]
        }

        questions_df = pd.DataFrame(dic)
        questions_df['Original_QType'] = questions_df['QType']
        mask = np.array(questions_df['QID'].str.contains('_TEXT'))
        questions_df.loc[mask, 'QType'] = 'TE'

        dic = {
            "QID": q_id_lt,
            "Q_Value": q_value,
            "Q_AnsId": q_ans_id
        }

        question_values = pd.DataFrame(dic)
        questions_df = clean_qs(questions_df)

        return questions_df, question_values


def dic_data_type(qdf, q_vals):
    """
    Determine and organize the data types for survey questions.
    """
    # Get lists of the Free Text columns
    mask = np.array(qdf['QType'] == 'TE') & np.array(qdf['Q_Numeric'] == False)
    free_text_cols = set(qdf.QID[mask])

    # Get lists of the FileUpload columns
    file_up_cols = set(qdf.QID[qdf['QType'] == 'FileUpload'])

    # Get list of Meta columns (essentially categorical but without predefined categories)
    meta_cols = set(qdf.QID[qdf['QType'] == 'Meta'])

    # Get lists of the Draw columns (often signatures)
    draw_cols = set(qdf.QID[qdf['QType'] == 'Draw'])

    # Frequency plots: Multiple choice questions
    mc = set(q_vals['QID'])

    # Timing columns (track page times)
    timing_cols = set(qdf.QID[qdf['QType'] == 'Timing'])

    # Date columns
    date_cols = set(qdf.QID[qdf['QType'] == 'SBS'])

    # Rank Order columns
    ro_cols = set(qdf.QID[qdf['QType'] == 'RO'])

    # Group columns list questions and the responses below
    group_cols = set(qdf.QID[qdf['QType'] == 'PGR'])

    # Get the numeric columns
    num_cols = set(qdf.QID[qdf['Q_Numeric']])

    # Create a dictionary of all the different column types
    col_data_types = {
        'MultipleChoice': list(mc),
        'Numeric': list(num_cols),
        'FreeText': list(free_text_cols),
        'RankOrder': list(ro_cols),
        'FileUpload': list(file_up_cols),
        'Group': list(group_cols),
        'MetaData': list(meta_cols),
        'Draw': list(draw_cols),
        'Timing': list(timing_cols),
        'Dates': list(date_cols)
    }

    # For each QID in QDF, determine the data type
    data_type = []
    for cid in qdf['QID']:
        if cid in col_data_types.get('MultipleChoice'):
            data_type.append('MultipleChoice')
        elif cid in col_data_types.get('Numeric'):
            data_type.append('Numeric')
        elif cid in col_data_types.get('FreeText'):
            data_type.append('FreeText')
        elif cid in col_data_types.get('RankOrder'):
            data_type.append('RankOrder')
        elif cid in col_data_types.get('FileUpload'):
            data_type.append('FileUpload')
        elif cid in col_data_types.get('Group'):
            data_type.append('Group')
        elif cid in col_data_types.get('MetaData'):
            data_type.append('MetaData')
        elif cid in col_data_types.get('Draw'):
            data_type.append('Draw')
        elif cid in col_data_types.get('Timing'):
            data_type.append('Timing')
        elif cid in col_data_types.get('Dates'):
            data_type.append('Dates')

    qdf['DataType'] = data_type
    return qdf


# yul1.qualtrics.com/API/v3/surveys/{surveyId}/permissions/collaborations
# 

# curl --request POST \
#   --url https://yul1.qualtrics.com/API/v3/surveys/surveyId/permissions/collaborations \
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
