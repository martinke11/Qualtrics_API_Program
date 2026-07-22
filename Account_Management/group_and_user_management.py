# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:58:03 2025

@author: Kieran Martin
"""
import json
import os
import QualAPI as qa
import requests
import re
import sys
from config import (
    set_project_directory,
    get_qualtrics_credentials_path
)

PROJECT_DIRECTORY = set_project_directory()
print("Working directory changed to:", PROJECT_DIRECTORY)

QUALTRICS_CREDENTIALS_PATH = get_qualtrics_credentials_path()
print("Qualtrics credentials path:", QUALTRICS_CREDENTIALS_PATH)
with open(QUALTRICS_CREDENTIALS_PATH) as f:
    qualtrics_creds = json.load(f)

###############################################################################
# Extract client ID, secret, and data center from credentials
client_id = creds.get('ID')
client_secret = creds.get('Secret')
data_center = creds.get('DataCenter')
base_url = f'https://{data_center}.qualtrics.com'

grant_type = 'client_credentials'
scope = ('write:mailing_lists read:users read:groups write:groups manage:groups '
        'manage:mailing_list_contacts manage:mailing_lists')
data = qa.return_kwargs_as_dict(grant_type=grant_type, scope=scope)

# Get the bearer token
bearer_token_response = qa.get_token(base_url, client_id, client_secret, data)
token = bearer_token_response.get('access_token')

users_list = qa.get_users(base_url, token)

group_list = qa.get_groups(base_url, token)

# Get group details
group_id = "GR_"  # existing group
details = qa.get_group_details(base_url, token, group_id)
print(json.dumps(details, indent=2))

# Create new group
group_name = ""
group_type = "GT_"
division_id = None
create_response = qa.create_group(base_url, token, group_name, group_type, division_id)
print(create_response)

# Add a Single User
group_id = "GR_"
user_id = "UR_"  # single user
response = qa.add_user_to_group(base_url, token, group_id, user_id)

print("Response from adding user:", response)

# Get list of users in a group
group_id = "GR_"
response = qa.list_users_in_group(
    base_url=base_url,
    token=token,
    group_id=group_id
)

print("Users in group:")
for user in response.get("result", {}).get("elements", []):
    print(f"- {user.get('firstName')} {user.get('lastName')} ({user.get('email')})")
