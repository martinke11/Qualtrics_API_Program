# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:14:09 2025

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
from io import BytesIO
import QualAPI as qa
import requests
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
from collections import Counter
import sys
PROJECT_DIRECTORY = os.chdir('C:\\Users\\484843\\Documents\\GitHub\\Qualtrics-API-Program')
if PROJECT_DIRECTORY not in sys.path:
    sys.path.append(PROJECT_DIRECTORY)
import QualAPI as qa
QUALTRICS_CREDENTIALS_PATH = os.path.expanduser(
    'C:\\Users\\484843\\Documents\\GitHub\\Qualtrics-API-Program\\copa_qualtrics_credentials.txt'
)
###############################################################################
# Load Qualtrics credentials from a JSON file using the defined path
with open(QUALTRICS_CREDENTIALS_PATH) as f:
    creds = json.load(f)

# Extract client ID, secret, and data center from credentials
client_id = creds.get('ID')
client_secret = creds.get('Secret')
data_center = creds.get('DataCenter')
base_url = f'https://{data_center}.qualtrics.com'

# Define survey name and set up parameters for token request
survey_name = "Test Survey"
grant_type = 'client_credentials'
scope = 'read:distributions write:libraries read:libraries read:surveys read:survey_responses write:distributions manage:distributions read:mailing_lists read:mailing_list_contacts write:mailing_lists read:users read:groups write:groups manage:groups manage:mailing_list_contacts manage:mailing_lists read:directories'
data = qa.return_kwargs_as_dict(grant_type=grant_type, scope=scope)

# Get the bearer token
bearer_token_response = qa.get_token(base_url, client_id, client_secret, data)
token = bearer_token_response.get('access_token')

# Retrieve the list of surveys and find the survey ID
survey_list_df = qa.get_survey_list(base_url, token)
survey_id = qa.get_survey_id_by_name(survey_list_df, survey_name)

libraries = qa.list_libraries(base_url, token)
library_id = "GR_9TttXzNhREpoOBE"
library_messages = qa.list_library_messages(base_url, token, library_id)
print(libraries)

# 'category' defines what the message will be used for. to send a message via email 
# and text there need to be 2 seperate messages
# Create an SMS invite message
sms_invite_messages = {
    "en": "Hello! This is a test SMS invite message. Please complete our survey."
}
response_sms_invite = qa.create_library_message(
    base_url=base_url,
    token=token,
    library_id=library_id,
    description="SMS Invite for Survey",  # A short description
    category="smsInvite",                 # Use smsInvite for SMS invites
    messages=sms_invite_messages
)
print("SMS Invite creation response:")
print(response_sms_invite)
text_message_id = "MS_0THibS4cn4ncROS"

# Create an Email invite message
email_invite_messages = {
    "en": "Hello! This is a test Email invite message. Please complete our survey."
}
response_email_invite = qa.create_library_message(
    base_url=base_url,
    token=token,
    library_id=library_id,
    description="Email Invite for Survey",  # A short description
    category="invite",                      # Use invite for Email invites
    messages=email_invite_messages
)
print("\nEmail Invite creation response:")
print(response_email_invite)
email_message_id = "MS_5cKXw0g1AlnRjam"

# Retrieve text message
response_text = qa.get_library_message(
    base_url=base_url,
    token=token,
    library_id=library_id,
    message_id=text_message_id
)

# Retrieve email message
response_email = qa.get_library_message(
    base_url=base_url,
    token=token,
    library_id=library_id,
    message_id=email_message_id
)

# Print both message responses
print("Text message response:")
print(response_text)
print("\nEmail message response:")
print(response_email)

# need functions for update library message and delete library message

# list distrobutions
distributions_response = qa.list_distributions(
    base_url=base_url,
    token=token,
    survey_id=survey_id,
#    distribution_request_type="Invite", # optional filter, omit to list all
    page_size=50,                        # optional custom page size
    use_new_pagination_scheme=True       # enable new pagination scheme
)

print(distributions_response)


directory_id = "POOL_DqlX1IukmQWE3JL"
mailing_list_id = "CG_2a8OvyFeLS6yQ0Z"

# Example usage: Send an Email invite
email_distribution_response = qa.create_distribution(
    base_url=base_url,
    token=token,                     # Bearer token with write:distributions scope
    message_library_id=library_id,   # e.g., "GR_9TttXzNhREpoOBE"
    message_id=email_message_id,     # e.g., "MS_5cKXw0g1AlnRjam"
    mailing_list_id=mailing_list_id, # e.g., "CG_2a8OvyFeLS6yQ0Z"
    directory_id=directory_id,       # e.g., "POOL_DqlX1IukmQWE3JL"
    survey_id=survey_id,             # Survey ID
    from_email="noreply@mydomain.com",
    reply_to_email="noreply@mydomain.com",
    from_name="Kieran's Test",
    subject="Please check if the survey link works",   # Subject line
    expiration_date="2025-12-31T23:59:59Z",
    scheduled_send_datetime="2025-01-31T08:00:00Z"  # Set your desired send datetime
)

print("Email distribution response:")
print(email_distribution_response)


# Example usage: Send an SMS invite
sms_distribution_response = qa.create_distribution(
    base_url="https://yul1.qualtrics.com",
    token=token,                     # Bearer token with write:distributions scope
    message_library_id=library_id,   # e.g., "GR_9TttXzNhREpoOBE"
    message_id=text_message_id,      # e.g., "MS_0THibS4cn4ncROS"
    mailing_list_id=mailing_list_id, # e.g., "CG_2a8OvyFeLS6yQ0Z"
    directory_id=directory_id,       # e.g., "POOL_DqlX1IukmQWE3JL"
    survey_id=survey_id,             # Survey ID
    from_email="noreply@mydomain.com",  # Still required in the payload
    reply_to_email="noreply@mydomain.com",
    from_name="Kieran's Test",             # Possibly displayed in SMS context
    subject="Please check if the survey link works", # Usually not shown in an SMS, but required
    expiration_date="2025-12-31T23:59:59Z",
    scheduled_send_datetime="2025-01-31T08:00:00Z"
)

print("\nSMS distribution response:")
print(sms_distribution_response)

















