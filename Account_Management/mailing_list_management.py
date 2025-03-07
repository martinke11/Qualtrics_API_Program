# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:05:39 2025

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
scope = 'write:distributions manage:distributions read:mailing_lists read:mailing_list_contacts write:mailing_lists read:users read:groups write:groups manage:groups manage:mailing_list_contacts manage:mailing_lists read:directories'
data = qa.return_kwargs_as_dict(grant_type=grant_type, scope=scope)

# Get the bearer token
bearer_token_response = qa.get_token(base_url, client_id, client_secret, data)
token = bearer_token_response.get('access_token')

directories_response = qa.list_directories(base_url, token)
print("Directories:", directories_response)

# Example variables: Define the directory ID for the mailing list then choose a name
directory_id = "POOL_DqlX1IukmQWE3JL"
new_mailing_list_name = "Findings Letter Complaint Survey Contact List"

# List mailing lists
list_mailing_list_response = qa.list_mailing_lists(base_url, token, directory_id)
print("Mailing Lists:", list_mailing_list_response)

# Create a mailing list using 'new_mailing_list_name'
create_mailing_list_response = qa.create_mailing_list(base_url, token, directory_id, new_mailing_list_name)
print("Created Mailing List:", create_mailing_list_response)

# after you create the mailing list it will give you the 'mailingListId' to use 
# to make any updates/edits to the list
mailing_list_id = "CG_3pqXzIq5nfuuoef"

# Get mailing list details
get_response = qa.get_mailing_list(base_url, token, directory_id, mailing_list_id)
print("Mailing List Details:", get_response)

# Share mailing list with user group in Qualtrics. can get group ID from 
# group_management.py code
prad_group_id = "GR_9TttXzNhREpoOBE"  
new_name = "Testing Mailing List - Shared with PRAD"
# Share the mailing list with the PRAD group
update_response = qa.update_mailing_list(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    name=new_name,
    owner_id=prad_group_id
)

print("Update Mailing List Response:", update_response)


# Update mailing list
update_response = qa.update_mailing_list(base_url, token, directory_id, mailing_list_id, "Updated Name")
print("Updated Mailing List:", update_response)

# Delete mailing list
delete_response = qa.delete_mailing_list(base_url, token, directory_id, mailing_list_id)
print("Deleted Mailing List:", delete_response)

# Managing Contact lists within a mailing list
# List contacts in the mailing list
response = qa.list_contacts_in_mailing_list(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    page_size=50,
    include_embedded=True  # This will now correctly pass as 'true'
)
print("Contacts in Mailing List:", response)

# definitions below may be reduntent unless you need to use a different list
directory_id = "POOL_DqlX1IukmQWE3JL"
mailing_list_id = "CG_3pqXzIq5nfuuoef"

# Create a new contact in the mailing list
create_response = qa.create_contact_in_mailing_list(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    first_name="Morgan",
    last_name="martin",
    email="morgan.mcguirk@cityofchicago.org",
    phone="2489821887",
    ext_ref="External125",
    embedded_data={"Department": "COPA", "Location": "Chicago"},
    private_embedded_data={"Salary": "Confidential"},
    language="en",
    unsubscribed=False
)
print("Created Contact Response:", create_response)


# List bounced contacts in the mailing list
bounced_response = qa.list_bounced_mailing_list_contacts(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    page_size=50,
    since="2023-01-01T00:00:00Z"  # Optional filter by date
)
print("Bounced Contacts:", bounced_response)

# List opted-out contacts in the mailing list
opted_out_response = qa.list_opted_out_mailing_list_contacts(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    page_size=50,
    since="2023-01-01T00:00:00Z"  # Optional filter by date
)
print("Opted-Out Contacts:", opted_out_response)


# Retrieve details of a specific contact in the mailing list
contact_id = "CID_012345678901234"

contact_response = qa.get_mailing_list_contact(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    contact_id=contact_id
)
print("Contact Details:", contact_response)


# Update details of a specific contact in the mailing list
update_response = qa.update_mailing_list_contact(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    contact_id=contact_id,
    firstName="Jane",
    lastName="Smith",
    email="jane.smith@example.com",
    phone="9876543210",
    extRef="UpdatedExternal123",
    embeddedData={"UpdatedField": "UpdatedValue"},
    privateEmbeddedData={"PrivateField": "ConfidentialUpdated"},
    unsubscribed=True
)
print("Updated Contact Response:", update_response)

# Contact ID to update
contact_id = "CID_1z7aPGr7f9brODQ"
new_last_name = "McGuirk"

update_contact_response = qa.update_mailing_list_contact(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    contact_id=contact_id,
    lastName=new_last_name
)

print("Update Contact Response:", update_contact_response)

# Example usage to update "Method of Contact Preference"
embedded_data_updates = {
    "CID_1z7aPGr7f9brODQ": {"Method of Contact Preference": "phone"},
    "CID_C79iaQhMlJFqYKt": {"Method of Contact Preference": "email"},
    "CID_3QRkZTwXOFHMSKv": {"Method of Contact Preference": "phone"}
}

for contact_id, embedded_data in embedded_data_updates.items():
    response = qa.update_mailing_list_contact(
        base_url=base_url,
        token=token,
        directory_id=directory_id,
        mailing_list_id=mailing_list_id,
        contact_id=contact_id,
        embeddedData=embedded_data
    )

print(f"Updated Contact {contact_id} Response:", response)
    
    
# Delete a specific contact from the mailing list
contact_id = "CID_25F3YXmkjmTFW82"
contact_id = "CID_29sHaG5IeRqFJHE"
contact_id = "CID_1lAagVjvFNvZNfB"


delete_response = qa.delete_contact_from_mailing_list(
    base_url=base_url,
    token=token,
    directory_id=directory_id,
    mailing_list_id=mailing_list_id,
    contact_id=contact_id
)
print("Delete Contact Response:", delete_response)




































