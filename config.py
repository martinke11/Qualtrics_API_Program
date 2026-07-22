# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:05:19 2025

@author: Kieran Martin
"""
import os
import json

def load_config(config_filename="config.json") -> dict:
    """
    Load and return the JSON config from ~/config.json
    """
    home_directory = os.path.expanduser("~")
    config_file_path = os.path.join(home_directory, config_filename)
    with open(config_file_path, "r") as f:
        return json.load(f)


def set_project_directory(config_filename="config.json") -> str:
    """
    Change into your Qualtrics‑API repo root (from config["qualtrics_api"]["root"])
    and return that path.
    """
    config = load_config(config_filename)
    PROJECT_DIRECTORY = os.path.expanduser(config["qualtrics_api"]["qualtrics_api_root"])
    os.chdir(PROJECT_DIRECTORY)
    return PROJECT_DIRECTORY


def get_sql_credentials_path(config_filename="config.json") -> str:
    """
    Expand and return the path to your SQL creds file
    as defined in your config.json.
    """
    config = load_config(config_filename)
    return os.path.expanduser(config["sql_credentials_path"])


def read_credentials(file_path):
    """
    Reads credentials from txt file in order for this python script to access the 
    data warehouse and execute the query

    Args:
        file_path(str): Path to credentials file on local
    Returns:
        dictionary(dict): the credentials file as a dictionary pulled into 
        python environment
    """
    creds = {}
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            key, value = line.split('=', 1)
            creds[key.strip()] = value.strip()
    return creds


def get_qualtrics_credentials_path(config_filename="config.json") -> str:
    """
    Return the expanded path to your Qualtrics creds JSON
    as defined in your config.json.
    """
    config = load_config(config_filename)
    return os.path.expanduser(config["qualtrics_api"]["qualtrics_credentials_path"])
