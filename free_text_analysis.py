# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:17:57 2024

@author: 484843
"""
###############################################################################
# Using the question_df, extract only those columns with questions
df = responses_df.loc[:, question_df['question_id'].tolist()]

# This code will find all the NaN values. It can be used to clean and determine NaN frequencies
nan_mask = df.isna()
keep_mask = np.array(nan_mask.sum(axis=1) < len(df.columns))

df = df.loc[keep_mask, :]
df = df.reset_index(drop=True)

###############################################################################
#nan_mask = df.isna()
missing_data = nan_mask.sum(axis=0).tolist()
present_data = len(df) - np.array(missing_data)

# Get the fraction of respondents by question
missing_data_fraction = np.array(missing_data) / len(df)
present_data_fraction = 1 - missing_data_fraction

# Create a DataFrame
missing_text_df = {
    'question_id': df.columns,
    "MissingDataN": missing_data,
    "PresentDataN": present_data,
    "MissingDataFrc": missing_data_fraction,
    "PresentDataFrc": present_data_fraction
}

missing_text_df = pd.DataFrame(missing_text_df)

free_text_columns = question_df.question_id[np.array(question_df.data_type == 'FreeText')]

# Grab the missing data for the text
mask_missing_text_df = np.array(np.isin(missing_text_df['question_id'], list(free_text_columns)))
mask_missing_text_df = missing_text_df.loc[mask_missing_text_df, :]

