#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:22:33 2024

@author: kieranmartin
"""
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches
from docx.shared import Pt
import requests
from docx.enum.text import WD_ALIGN_PARAGRAPH
from collections import Counter
import requests
from googletrans import Translator
import json
import os
import time
#import emoji
import warnings
import httpx
import nltk
# nltk.download('all') # nltk.download('vader_lexicon')
from textblob import TextBlob
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
pd.options.mode.chained_assignment = None 

# Ensure that all output is printed without truncation
pd.set_option('display.max_colwidth', None)  # Do not truncate column content
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

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
    
# build out from here 
