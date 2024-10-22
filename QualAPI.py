# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:50:53 2023

@author: jhanley
"""

# Qualtrics API Functions
# Reference for Qualtrics API endpoints:
# https://api.qualtrics.com/0f8fac59d1995-api-reference

import requests
import pandas as pd
import numpy as np
import re


# Build a function to build out the data argument
# Convert keyword arguments to a string
def convert_kwargs_to_string(**kwargs): # formerly DefData
    # Convert to a string
    r = str(kwargs)
    return r

# Format a dictionary for JSON-like representation
def format_dict_for_json(dic): # formally DefData1
    # Some data formats in Qulatrics are different. So need to change the 
    # format
    # Input must be a dictionary.
    # First convert to a string
    r = str(dic)
    # Now change the quotes around
    r = r.replace("'", '"')
    return r

def return_kwargs_as_dict(**kwargs): # formally DefDataT
    # Convert to a dictionary
    r = kwargs
    return r

# API functions
# Get a token that will last an hour
def get_token(base_url, client_id, client_secret, data):
    # list of scopes:
    # manage:activity_logs
    # manage:all
    # manage:contact_frequency_rules
    # manage:contact_transactions
    # manage:customer_data_requests
    # manage:directories
    # manage:directory_contacts
    # manage:distributions
    # manage:divisions
    # manage:erasure_requests
    # manage:groups
    # manage:libraries
    # manage:mailing_list_contacts
    # manage:mailing_lists
    # manage:organizations
    # manage:participants
    # manage:samples
    manage:subscriptions
    manage:survey_responses
    manage:survey_sessions
    manage:surveys
    # manage:tickets
    # manage:users
    # openid
    # profile
    # read:activity_logs
    # read:contact_frequency_rules
    # read:contact_transactions
    # read:directories
    # read:directory_contacts
    # read:distributions
    # read:divisions
    # read:groups
    # read:imported_data_projects
    # read:libraries
    # read:mailing_list_contacts
    # read:mailing_lists
    # read:organizations
    # read:participants
    # read:samples
    read:subscriptions
    read:survey_responses
    read:survey_sessions
    read:surveys
    # read:tickets
    # read:users
    # write:automations
    # write:contact_frequency_rules
    # write:contact_transactions
    # write:directory_contacts
    # write:distributions
    # write:divisions
    # write:embedded_dashboards
    # write:embedded_xid_profile_cards
    # write:groups
    # write:imported_data_projects
    # write:libraries
    # write:mailing_list_contacts
    # write:mailing_lists
    # write:participants
    # write:samples
    write:subscriptions
    write:survey_responses
    write:survey_sessions
    write:surveys
    # write:tickets
    # write:users
    tknURL = base_url + "/oauth2/token"
    
    r = requests.post(tknURL, auth=(client_id, client_secret), data=data)
    
    return r.json()

# Get a list of Users
def get_users(base_url, tkn):
    # set the url
    s = '{0}/API/v3/users'.format(base_url)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn}) 
    # convert the data into a more readable format
    r = r.json()    
    return r

# get a surveys metadata
def get_survey_meta(base_url, tkn, SurveyId):
    # set the url
    s = '{0}/API/v3/survey-definitions/{1}/metadata'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn}) 
    # convert the data into a more readable format
    r = r.json()    
    return r

# Change a survey's metadata
def change_survey_meta(base_url, tkn, SurveyId, data):
    # set the url
    s = '{0}/API/v3/survey-definitions/{1}/metadata'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.put(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn},
                     data=data)
    # convert the data into a more readable format
    r = r.json()    
    return r

# Update a survey
def update_survey(base_url, tkn, SurveyId, data):
    # set the url
    s = '{0}/API/v3/surveys/{1}'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.put(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn},
                     data=data)
    # convert the data into a more readable format
    r = r.json()    
    return r

# get the survey responses
def get_survey_responses(base_url, tkn, SurveyId, fid):
    # set the url
    s = '{0}/API/v3/surveys/{1}/export-responses/{2}/file'.format(base_url, SurveyId, fid)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # convert the data into a more readable format
    #r = r.json()
    return r 

# Pull the survey list
def get_survey_list(base_url, tkn):
    # extract all the surveys and their ids
    # Set Next Page to Data Type None Flag to True 
    Flag = True
    
    # Set a counter
    count = 0
    
    # Set cnum to blank
    cnum = ""
    
    while Flag:
        # pull the data
        if count == 0:
            # set the url
            s = '{0}/API/v3/surveys'.format(base_url)
        else:
            # set the url
            s = '{0}/API/v3/surveys?offset={1}'.format(base_url, cnum)
        
        # Pull the surveys
        r = requests.get(s, headers={"Authorization": "Bearer " + tkn})
        # convert the data into a more readable format
        r = r.json()

        # Extract the Last of the Survey List
        CurDF = pd.DataFrame(r.get("result").get("elements"))
        
        if count == 0:
            # There is just one page of results
            SurveyDF = CurDF
        else: 
            # Else add to the current dataframe
            SurveyDF = pd.concat([SurveyDF, CurDF])
            
        if r.get("result").get("nextPage") is None:
            # Then set the Flag to False to end the loop
            Flag = False
            
        else:
            # Then the flag is still true
            # Extract the Next Page string
            NPstr = r.get("result").get("nextPage")
            # Extract the current number for offset
            cnum = NPstr.split("?offset=",1)[1]        
            # update the counter
            count = count + 1
    # reset the indices
    SurveyDF = SurveyDF.reset_index(drop = True)
    return SurveyDF

# Pull survey questions
def getSurveyQs(base_url, tkn, SurveyId):
    #set the URL
    s = '{0}/API/v3/surveys/{1}'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.get(s, headers={"Authorization": "Bearer " + tkn})
    # Convert the data into a more readable format
    surveyq_dict = r.json()
    
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
    if 'result' in surveyq_dict and 'questions' in surveyq_dict['result']:
        for question_id, question_data in surveyq_dict['result']['questions'].items():
            if 'questionText' in question_data:
                question_data['questionText'] = strip_html(question_data['questionText'])
            if 'choices' in question_data:
                question_data['choices'] = strip_html(question_data['choices'])

    return surveyq_dict
    

# Pulls a little more information than the survey Qs
def getFullSurvey(base_url, tkn, SurveyId):
    # set the url
    s = '{0}/API/v3/survey-definitions/{1}'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn}) 
    # convert the data into a more readable format
    r = r.json()    
    return r

# Post the survey Response Export
def getSurveyRX(base_url, tkn, SurveyId):
    # set the url
    s = '{0}/API/v3/surveys/{1}/export-responses'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.post(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn},
                     data='{"format": "json", "compress": false}')
    # convert the data into a more readable format
    r = r.json()    
    return r

# Function to get the Response Export Progress
def getSurveyRXp(base_url, tkn, SurveyId, EPid):
    # set the url
    s = '{0}/API/v3/surveys/{1}/export-responses/{2}'.format(base_url, SurveyId, EPid)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # convert the data into a more readable format
    r = r.json()    
    return r



def getGroups(base_url, tkn):
    # set the url
    s = '{0}/API/v3/groups'.format(base_url)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # convert the data into a more readable format
    r = r.json()
    return r

# Share a Survey with a group or person
def postCollabs(base_url, tkn, SurveyId, data):
    # set the url
    s = '{0}/API/v3/surveys/{1}/permissions/collaborations'.format(base_url, SurveyId)
    # Pull the survey data
    r = requests.post(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn},
                     data=data)
    # convert the data into a more readable format
    r = r.json()    
    return r


# Share a Survey with a group or person
def getWhoAmI(base_url, tkn): 
    # set the url
    s = '{0}/API/v3/whoami'.format(base_url)
    # Pull the survey data
    r = requests.get(s, 
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + tkn})
    # convert the data into a more readable format
    r = r.json()    
    return r

REcollabJson = open('RandE_Collaborator.json')

def OrganizeResponses(Responses):
    # DF = Dataframe of the Responses
    # for each response extract and put in a dataFrame
    DF = []
    for j in Responses:
        # Extract the current response and normalize
        tDF = pd.json_normalize(j.get('values'))
        if len(DF) == 0:
            # Then tDF = DF
            DF = tDF
        else:
            # Then concatenate
            DF = pd.concat([DF, tDF])
            
    # Reset the DF index
    DF = DF.reset_index(drop = True)

    # Reorder the columns so that they appear in order
    # Find the QID columns 
    ColNames = DF.columns
    QCols = [x for x in ColNames if x.startswith('QID')]
    # NonQcols
    NonQCols = list(set(ColNames) - set(QCols))
    DFTemp1 = DF.loc[:, np.isin(ColNames, NonQCols)]
    
    # Create a function to order the column headers
    Q_Num = []
    Q_SubQ = []
    Q_LoopN = []
    
    for q in QCols:
        # Split the QID
        CurSpl = re.split('_|#', q) 
        # Find the QID index
        QIDind = [i for i in range(len(CurSpl)) if 'QID' in CurSpl[i]]
        # Get the max index
        MaxInd = len(CurSpl) - 1
        
        # Determine where the QID is 
        if QIDind[0] == 0:
            # Then this is not a loop question
            Q_LoopN.append(0)
        elif QIDind[0] == 1:
            # Then this is a loop question
            Q_LoopN.append(int(CurSpl[0]))
        else:
            print('Issues with ' + q)
            break
        # Grab the Question
        QSpl = re.split('QID', CurSpl[QIDind[0]])
        # Append the number
        Q_Num.append(int(QSpl[1]))
        # Determine if there is a subquestion
        if MaxInd > QIDind[0]:
            # Then there could be a sub question... check
            if CurSpl[QIDind[0] + 1].isnumeric():
                # Then get the sub question
                Q_SubQ.append(int(CurSpl[QIDind[0] + 1]))
            else:
                # there is no subquestion
                Q_SubQ.append(0)
        else:
            # the is no subquestion
            Q_SubQ.append(0)
    
    # Create a data frame
    tdic = {'QCol': QCols,
            'Q_Num': Q_Num,
            'Q_SubQ': Q_SubQ,
            'Q_LoopN': Q_LoopN}
    QDF = pd.DataFrame(tdic)
    QDF = QDF.sort_values(by = ['Q_Num', 'Q_SubQ', 'Q_LoopN'], ascending = [True, True, True])
    
    # Get the Second DF
    DFTemp2 = DF.loc[:, QDF['QCol']]
    
    # Concatenate
    DFN = pd.concat([DFTemp1, DFTemp2], axis = 1)
    
    return DFN

def CleanQs(QuestionsDF):
    # Clean the questions DF
    # Set the pattern that contains superflous text formating
    pat = r'(?=\<).+?(?<=\>)'
    for i, rows in QuestionsDF.iterrows():
        # Grab the current text
        CurT = QuestionsDF.loc[i, 'QText']
        # Remove superfluous text
        NewT = re.sub(pat, "", CurT)
        # Now do the same for any other patterns that don't translate well
        if '&#39;' in NewT:
            # This is an apostrophe
            pat1 = '&#39;'
            NewT = re.sub(pat1, "'", NewT)
        if '\n' in NewT:
            pat2 = '\n'
            NewT = re.sub(pat2, "", NewT)
        # Now Fix the QText
        QuestionsDF.loc[i, 'QText'] = NewT
    return QuestionsDF
        
def ExtractColumnDataTypes(QDic, RDF, base_url, tkn, SurveyId):
    # pull the column names 
    ColNames = RDF.columns
    # Find the columns that start with QID
    QCols = [x for x in ColNames if 'QID' in x]
    
    # Check to see if there are any questions
    if len(QCols) > 0:
    
        # Replace _Text since that will not map
        # QCols_MinT = [x.replace('_TEXT', '') for x in QCols]
    
        # Extract the Question text and Question type, Question value and Question ID
        QID = []
        QName = []
        QText = []
        QType = []
        QTypeSel = []
        QID_LT = []
        Q_Value = []
        Q_AnsId = []
        Q_Numeric = []
        QKeep = []
        
        # Create a different set for groups
        QG_QID = []
        QG_Value = []
        QG_AnsId = []
        
        # Get list of survey Question Keys
        KeyList = list(QDic.keys())                
        
        for x in QCols:
            # Split the current QID
            CurSpl = re.split('_|#', x) # The # symbol may indicate that the variable is numeric
            # Find the QID index
            QIDind = [i for i in range(len(CurSpl)) if 'QID' in CurSpl[i]]
            
            # check to see if the key is present
            if CurSpl[QIDind[0]] in KeyList:
                # Record the QID col
                QID.append(x)
                # Pull the question key
                CurQ = QDic.get(CurSpl[QIDind[0]])
                # Record the question name
                QName.append(CurQ.get('questionName'))
                # Get the question type
                CurType = CurQ.get('questionType').get('type')
                QType.append(CurType)
                CurTypeSel = CurQ.get('questionType').get('selector')
                QTypeSel.append(CurTypeSel)
                
                # Check to see if the type is numeric
                # Check to see if the data is there
                if 'validation' in CurQ:
                    # Then check to see if validation type is there
                    CurValid = CurQ.get('validation')
                    if 'type' in CurValid:
                        # Then check to is if it is numeric
                        CurValidType = CurValid.get('type')
                        if 'ValidNumber' in CurValidType:
                            # Then the value is numeric
                            Q_Numeric.append(True)
                        else:
                            # Value is not numeric
                            Q_Numeric.append(False)
                    else:
                        # Value is not numeric
                        Q_Numeric.append(False)                
                else:
                    # Value is not numeric
                    Q_Numeric.append(False)
                
                if CurType == 'Matrix':
                    # Then there are sub questions
                    # Get the first half of the question 
                    QT1 = CurQ.get('questionText')
                    # Now need to dig deeper and pull the second half the question
                    QT2 = CurQ.get('subQuestions').get(CurSpl[QIDind[0]+1]).get('choiceText')
                    # Get the Question Text (marking the split with a bar)
                    QT = QT1 + '| ' + QT2
                    QText.append(QT)
                    QKeep.append(True)
                    # Pull the values
                    Scores = CurQ.get('subQuestions').get(CurSpl[QIDind[0]+1]).get('scoring')
                    # Check to see if Scores is None
                    # Check to see if the subType is CS
                    if CurTypeSel == 'CS':
                        # The data is numeric
                        Q_Numeric[-1] = True
                    else:
                        if Scores is None:
                            # Then record the Choices
                            Choices = CurQ.get('choices')
                            for xx in Choices:
                                # Get the Current Data
                                CurChoice = Choices.get(xx)
                                # Record the QID_MC_LT
                                QID_LT.append(x)
                                # Record the answer ID
                                Q_AnsId.append(CurChoice.get('recode'))
                                # Record the answer value
                                Q_Value.append(CurChoice.get('choiceText'))                        
                        else:                        
                            # Then go to scoring
                            for xx in Scores:
                                # Record the question
                                QID_LT.append(x)
                                # Get the value
                                Q_Value.append(xx.get('value'))
                                # Get the answer Id and see if it exists
                                if xx.get('answerId') is None:
                                    # Then enter the value for AnswerId
                                    Q_AnsId.append(xx.get('value'))
                                else: 
                                    # There is a value so add it
                                    Q_AnsId.append(xx.get('answerId'))
                elif CurType == 'CS':
                    # the data type is cumulative sum, so will need to get subquestions
                    # Also reset the value to numeric
                    Q_Numeric[-1] = True                
                    # Then there are sub questions
                    # Get the first half of the question 
                    QT1 = CurQ.get('questionText')
                    # Now need to get each sub question
                    # Extract the Choices
                    Choices = CurQ.get('choices')
                    # Now need to dig deeper and pull the second half the question
                    QT2 = Choices.get(CurSpl[QIDind[0]+1]).get('choiceText')
                    # Get the Question Text (marking the split with a bar)
                    QT = QT1 + '| ' + QT2
                    QText.append(QT)
                    QKeep.append(True)
                elif CurType == 'RO':
                    # Then might need to get the full survey to map properly
                    if CurSpl[len(CurSpl) - 1] == 'TEXT':
                        # Then change the data type to text
                        QType[-1] = 'TE'
                        QTypeSel[-1] = 'TE'
                        # the data type is rank order, so will need to get subquestions
                        # Then there are sub questions
                        # Get the first half of the question 
                        QT = CurQ.get('questionText')
                        QText.append(QT)
                        QKeep.append(True)                            
                    else:
                        # Get the mapping
                        FSurv = getFullSurvey(base_url, tkn, SurveyId)
                        # Pull the results
                        CO = FSurv.get('result').get('Questions').get(CurSpl[QIDind[0]]).get('ChoiceOrder')
                        # Now Index the substring and subtract 1
                        CurKey = str(CO[int(CurSpl[QIDind[0]+1]) - 1])
                        # the data type is rank order, so will need to get subquestions
                        # Then there are sub questions
                        # Get the first half of the question 
                        QT1 = CurQ.get('questionText')
                        # Now need to get each sub question
                        # Extract the Choices
                        Choices = CurQ.get('choices')
                        # Now need to dig deeper and pull the second half the question
                        QT2 = Choices.get(CurKey).get('choiceText')
                        # Get the Question Text (marking the split with a bar)
                        QT = QT1 + '| ' + QT2
                        QText.append(QT)
                        QKeep.append(True)
                elif CurType == 'Slider':
                    # The data is slider and numeric
                    # Also reset the value to numeric
                    Q_Numeric[-1] = True                
                    # Then there are sub questions
                    # Get the first half of the question 
                    QT1 = CurQ.get('questionText')
                    # Now need to get each sub question
                    # Extract the Choices
                    Choices = CurQ.get('choices')
                    # Now need to dig deeper and pull the second half the question
                    QT2 = Choices.get(CurSpl[QIDind[0]+1]).get('choiceText')
                    # Get the Question Text (marking the split with a bar)
                    QT = QT1 + '| ' + QT2
                    QText.append(QT)
                    QKeep.append(True)
                elif CurType == 'Timing':
                    # Then the question is a timing question
                    # Reset the value to numeric
                    Q_Numeric[-1] = True   
                    # Get the question text
                    QText.append(CurQ.get('questionText'))
                    QKeep.append(True)
                elif CurType == 'SS':
                    # Then the question is a graphic slider
                    # Not all graphic sliders are numeric scale,
                    # I am assuming typesel is numeric
                    if CurTypeSel == 'TA':
                        # the data is numeric so adjust
                        # Reset the value to numeric
                        Q_Numeric[-1] = True   
                        # Get the question text
                        QText.append(CurQ.get('questionText'))
                        QKeep.append(True)
                    else:
                        print('Problems with Type SS (Graphical Slider)!!')
                elif CurType == 'PGR':
                    if CurSpl[len(CurSpl) - 1] == 'GROUP':
                        # Then there is a sub question
                        # Get the first half of the question 
                        QT1 = CurQ.get('questionText')
                        # Now need to get each sub question
                        # Extract the Choices
                        Groups = CurQ.get('groups')
                        # Now need to dig deeper and pull the second half the question
                        QT2 = Groups.get(CurSpl[QIDind[0]+1]).get('description')
                        # Get the Question Text (marking the split with a bar)
                        QT = QT1 + '| ' + QT2
                        QText.append(QT)
                        # Record the Values
                        QKeep.append(True)
                        # Extract the values
                        Items = CurQ.get('items')
                        for xx in Items:
                            CurItem = Items.get(xx)
                            # Get the QG_QID
                            QG_QID.append(x)
                            QG_AnsId.append(xx)
                            QG_Value.append(CurItem.get('description'))   
                    elif CurSpl[len(CurSpl) - 1] == 'RANK':
                        # then this is just the rank order of items
                        # The format is: 
                            # 1) QID
                            # 2) G followed by subquestion # (i.e., description recode)
                            # 3) The item key
                            # 4) the word RANK
                            # This is repeated information since the vectors already have the items listed in order
                            # No need to save this
                            QText.append(CurQ.get('questionText'))
                            QKeep.append(False)
                    elif CurSpl[len(CurSpl) - 1] == 'TEXT':
                        # then this is free text for the free text question
                        # Get the first half of the question 
                        QT1 = CurQ.get('questionText')
                        # Now need to get each sub question
                        # Extract the items
                        Items = CurQ.get('items')
                        # Get the item 
                        CurItem = Items.get(CurSpl[QIDind[0]+1])
                        # Now need to dig deeper and pull the second half the question
                        QT2 = CurItem.get('description')
                        # Get the Question Text (marking the split with a bar)
                        QT = QT1 + '| ' + QT2
                        QText.append(QT)
                        # Record the Values
                        QKeep.append(True)
                        # Reset the data type to text
                        QType[-1] = 'TE'
                        QTypeSel[-1] = 'TE'
                    else: 
                        # There is no subquestion
                        QText.append(CurQ.get('questionText'))
                        # Record the Values
                        QKeep.append(True)
                        # Extract the values
                        Items = CurQ.get('items')
                        for xx in Items:
                            CurItem = Items.get(xx)
                            # Get the QG_QID
                            QG_QID.append(x)
                            QG_AnsId.append(xx)
                            QG_Value.append(CurItem.get('description'))    
                else:
                    # Get the question text
                    QText.append(CurQ.get('questionText'))
                    # Keep the question
                    QKeep.append(True)
                    # If the question type is Multiple Choice Record the values
                    if CurQ.get('questionType').get('type') == 'MC':
                        if CurSpl[-1] == 'TEXT':
                            # Then the data is text
                            QType[-1] = 'TE'
                            QTypeSel[-1] = 'TE'
                        else:
                            # Then record the Choices
                            Choices = CurQ.get('choices')
                            for xx in Choices:
                                # Get the Current Data
                                CurChoice = Choices.get(xx)
                                # Record the QID_MC_LT
                                QID_LT.append(x)
                                # Record the answer ID
                                Q_AnsId.append(CurChoice.get('recode'))
                                # Record the answer value
                                Q_Value.append(CurChoice.get('choiceText'))
                    
        # convert to numpy arrays
        QID = np.array(QID)
        QName = np.array(QName)
        QText = np.array(QText)
        QType = np.array(QType)
        QTypeSel = np.array(QTypeSel)
        Q_Numeric = np.array(Q_Numeric)
        # Create dataframes
        dic = {"QID": QID[np.array(QKeep)],
               "QName": QName[np.array(QKeep)],
               "QText": QText[np.array(QKeep)],
               "QType": QType[np.array(QKeep)],
               "QTypeSel": QTypeSel[np.array(QKeep)],
               "Q_Numeric": Q_Numeric[np.array(QKeep)]}
        # Data frame with column headers and questions
        QuestionsDF = pd.DataFrame(dic)
        
        # Clean some of the Data types which are really text
        # Add a column to account for subquestions other
        QuestionsDF['Original_QType'] = QuestionsDF['QType']
        Mask = np.array(QuestionsDF['QID'].str.contains('_TEXT'))
        QuestionsDF.loc[Mask, 'QType'] = 'TE'
        
        dic = {"QID": QID_LT,
               "Q_Value": Q_Value,
               "Q_AnsId": Q_AnsId}
        # Data frame of the values to questions
        QuestionValues = pd.DataFrame(dic)
        
        QuestionsDF = CleanQs(QuestionsDF)
        
        return QuestionsDF, QuestionValues
    
    
def DicDataType(QDF, QVals):
    ###############################################################################
    
    # I am just going to do some partial coding. Please incorporate where you see 
    # fit in your code. We can meet if you have any questions
    
    # Go through the questions in order 
    # QDF['QTypeSel'] == 'SL' Is free text 
    # QDF['QTypeSel'] == 'FileUpload' Is info on files being uploaded
    
    # Get lists of the Free Text columns
    Mask = np.array(QDF['QType'] == 'TE') &\
        np.array(QDF['Q_Numeric'] == False)
    FreeTextCols = set(QDF.QID[Mask])
    
    # Get lists of the FileUpload columns
    FileUpCols = set(QDF.QID[QDF['QType'] == 'FileUpload'])
    
    # Get list of Meta columns 
    # These are essentially categorical but without predefined categories
    MetaCols = set(QDF.QID[QDF['QType'] == 'Meta'])
    
    # Get lists of the Draw columns (often signatures)
    DrawCols = set(QDF.QID[QDF['QType'] == 'Draw'])
    
    # Frequency Plots
    MC = set(QVals['QID'])
    
    # What to do with Timing columns... they track page times...
    # Nothing for now
    TimingCols = set(QDF.QID[QDF['QType'] == 'Timing'])
    
    # What to do with date columns
    DateCols = set(QDF.QID[QDF['QType'] == 'SBS'])
    
    # What to do with Rank Order columns? Heat Matrix?
    ROCols = set(QDF.QID[QDF['QType'] == 'RO'])
    
    # What to do with Group cols list Questions and the responses below
    GroupCols = set(QDF.QID[QDF['QType'] == 'PGR'])
    
    # This should get the numeric Cols
    NumCols = set(QDF.QID[QDF['Q_Numeric']])

    # Create a dictionary of all the different column types
    ColDataTypes = {'MultipleChoice': list(MC),
                    'Numeric': list(NumCols),
                    'FreeText': list(FreeTextCols),
                    'RankOrder': list(ROCols),
                    'FileUpload': list(FileUpCols),
                    'Group': list(GroupCols),
                    'MetaData': list(MetaCols),
                    'Draw': list(DrawCols),
                    'Timing': list(TimingCols),
                    'Dates': list(DateCols)}
    
    # For each QDF determine the data Type
    DataType = []
    for cid in QDF['QID']:
        if cid in ColDataTypes.get('MultipleChoice'):
            DataType.append('MultipleChoice')
        elif cid in ColDataTypes.get('Numeric'):
            DataType.append('Numeric')
        elif cid in ColDataTypes.get('FreeText'):
            DataType.append('FreeText')
        elif cid in ColDataTypes.get('RankOrder'):
            DataType.append('RankOrder')
        elif cid in ColDataTypes.get('FileUpload'):
            DataType.append('FileUpload')
        elif cid in ColDataTypes.get('Group'):
            DataType.append('Group')
        elif cid in ColDataTypes.get('MetaData'):
            DataType.append('MetaData')
        elif cid in ColDataTypes.get('Draw'):
            DataType.append('Draw')
        elif cid in ColDataTypes.get('Timing'):
            DataType.append('Timing')
        elif cid in ColDataTypes.get('Dates'):
            DataType.append('Dates')
            
    QDF['DataType'] = DataType
    return QDF

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
