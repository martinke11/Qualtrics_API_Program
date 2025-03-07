# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:58:03 2025

@author: 484843
"""
import json
import os
import QualAPI as qa
import requests
import re

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

grant_type = 'client_credentials'
scope = 'write:mailing_lists read:users read:groups write:groups manage:groups manage:mailing_list_contacts manage:mailing_lists'
data = qa.return_kwargs_as_dict(grant_type=grant_type, scope=scope)

# Get the bearer token
bearer_token_response = qa.get_token(base_url, client_id, client_secret, data)
token = bearer_token_response.get('access_token')

users_list = qa.get_users(base_url, token)

group_list = qa.get_groups(base_url, token)


# Get group details
group_id = "GR_9TttXzNhREpoOBE"  # existing group
details = qa.get_group_details(base_url, token, group_id)
print(json.dumps(details, indent=2))


# Create new group
group_name = "PRAD"
group_type = "GT_00AHzWl9mQEsnLD"
division_id = None
#create_response = qa.create_group(base_url, token, group_name, group_type, division_id)
#print(create_response)


# Add a Single User
group_id = "GR_9TttXzNhREpoOBE"
user_id = "UR_22zFQt6fXImwBpz"  # single user
response = qa.add_user_to_group(base_url, token, group_id, user_id)
print("Response from adding user:", response)


# Get list of users in a group
group_id = "GR_9TttXzNhREpoOBE"

response = qa.list_users_in_group(
    base_url=base_url,
    token=token,
    group_id=group_id
)

print("Users in group:")
for user in response.get("result", {}).get("elements", []):
    print(f"- {user.get('firstName')} {user.get('lastName')} ({user.get('email')})")






    

