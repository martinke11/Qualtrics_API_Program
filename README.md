# Qualtrics-API-Program
Repository for managing all scripts associated with using the Qualtrics API and generating reports from it.

### Managing API Credentials/Sensitive Information
- API Credentials(ClientID, Client Secret, and Data Center) must be stored locally in a plain text(.txt) file and added to git-ignore to prevent sensitive info from being shared/leaked.
- Any other sensitive information (surey names, names, locations, etc) may not be hardcoded into scripts in Repo. They can be used when running the scripts locally but if any scripts need to be edited sensitive information must be redcted prior.

### Version Control
[requirements.txt](https://github.com/martinke11/Qualtrics_API_Program/blob/main/requirements.txt) hold all of the library/package versions that the program currently runs on. To prevent bugs from version conflicts activate virtual environment to run Program:
1) Open Anaconda Promtp and navigate to repo:<br />
(base) C:\Your\Path>**cd C:\Your\Path\GitHub\Qualtrics-API-Program** (Replace with your file path)
3) Activate correct python version:<br /> 
(base) C:\Your\Path\\GitHub\Qualtrics_API_Program>**conda create --name my_env python=3.12**<br />
Proceed ([y]/n)?**y**<br />
(base) C:\Your\Path\\GitHub\Qualtrics_API_Program>**conda activate my_env**
4) The command lines should now start with (my_env). From there install the correct versions into the virtual environment:<br />
(my_env) C:\Your\Path\\GitHub\Qualtrics_API_Program>**pip install -r requirements.txt**
5) Make sure to deavctivate virtul enivonment when done running scripts. You should see 'my_env' turn back to 'base':<br />
(my_env) C:\Your\Path\GitHub\Qualtrics_API_Program>**conda deactivate**<br />
(base) C:\Your\Path\Documents\GitHub\Qualtrics_API_Program>

### Navigation
[QualAPI.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/QualAPI.py) is a py-module that stores all functions that need to be used for various tasks accross various scripts. This includes generating the token, pulling the list of surveys, and creating data frames required to analyze the survey responses. The final data frame is responses_df and will be used accross the script for anaylsis, where each script will make further transformations to responses_df depending on the task. Functions that only need to be run within 1 script(for a specific task after responses_df is loaded and cleaned) should be kept within that script. For example, the function translate_seperate_text_df in [translation.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/translation.py) is used for translation only and should be kept in the script doing the translation.

import statement: <br /> import QualAPI as qa
<br /> 

#### Frequency & Count Report: [frequency_count_report.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/frequency_count_report.py)
This script generates a report in word doc that includes:
1) A table for every quantitative question with the Question answer value, the assinged question answer values code (i.e. yes = 1, no = 2, for visualization purposes in case answer options are too long and dont present well in the chart), the count, and the frequency of the answers. 
2) Corresponding bar chart visualizations for the frequency distrobutions. Currently displays % on bars but can be updated to present count or even % and count together by updateing lines 247 - 259.
3) Page break between each question.

You will need to add a folder titled 'Reports' to the repo locally on your computer and then add it to .gitignore. This folder is where the output from the word doc is placed. It can also be used for any other output documentation from any of the other scripts.

NULL Included in analysis and visualizations: Because, in many cases, respondents skip certain questions, NULL responses are kept for each question to prevent confusion around inconsistant total responses across questions in a survey. Including NULL can also provide insight to which questions could be reworked/edited to be clearer/easier for the respondant (if NULL is particuarily high for a given question.) NULL responses are visualized with a red bar instead of blue. 

Current Version makes tables and plots of Multiple Choice Questions: Single Choice, Multiple Choice, Dropdown Lists, Random Choice, Matrix, and Numeric Questions: Rank Order, Sliders
- Not set up to do Side by Side questions(SBS/SBSMatrix), which should be avoided.
- Text Questions are not plotted here
- For Slider questions that use the Star visuals, surveys should be made with the default 5 stars, currently there is no way to automatically identify how many Stars were set in the survey. If a survey has a different about of stars, line 766 (slider_range = range(1, 6)) in the handle_slider_question function in [QualAPI.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/QualAPI.py) will need to be updated manually.

#### Pre & Post Survey Frequency Count Report: [pre_post_matching_and_analysis_with_same_survey.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/pre_post_matching_and_analysis_with_same_survey.py)<br>
This script does the same analysis as [frequency_count_report.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/frequency_count_report.py) but for pre and post data that was collected in the same survey with no unique ids. So the script first uses fuzzy matching to match pre and post responses, assigns a unique ID to each match, and then generates a report via word document.

#### Pre & Post General Analysis: [pre_post_general_analysis.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/pre_post_general_analysis.py)
This script houses buildible code for pre and post analysis where pre and post data was collected in two seperate surveys, or the same survey with pre and post data defined by a cutt-off date, or the same survey with fuzzy matching on names or other attributes to determine pre and post responses.

#### Translation Script: [translation.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/translation.py)
[Google Translate library documentation](https://libraries.io/pypi/googletrans/4.0.0rc1)
- googletrans 4.0.0rc1 is the best *free* api tool for translating data
  
There are 2 methods for transcribing a data frame:
1) translate_seperate_text_df function keeps the original untranslated column(s) and adds the new/translated column(s) next to it. Can be used for comparison purposes or checking with a native speaker.
2) translate_replace_full_df function replaces the original untranslatted column(s) with the translated column(s).

Currently this script end with optional code to export the translated version as a csv file. However, this script can be used as a py-module to import the translated data frame into nlp_analysis.py script for further NLP analysis.

#### Free Text Compliance Script: [free_text_compliance.py](https://github.com/martinke11/Qualtrics_API_Program/blob/main/free_text_compliance.py)
Gets a count + percentage of how many respondents completed each free text question. Useful for when deciding to do any NLP analysis by understanding the sample size i.e. if only 10 or 20 (or whatever number depending on the project) responses completed a free text question its likely not worth pursuing NLP analysis. 

