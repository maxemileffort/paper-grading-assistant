import shutil, time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

from utils import *

# Config
set_config()

# State
states = ['grading', 
          'uploaded_file', 
          'pts_possible', 
          'models_loaded', 
          'zipped_files',
          'grading_progress',
          'finished_grading',
          'grades_accepted',
          'grades_df',
          'grades_csv',
          'grade_review'
          ]

for state in states:
    if state not in st.session_state:
        st.session_state[state] = False

# State managers
def reset_state():
    for state in states:
        st.session_state[state] = False
        try:
            os.remove('./papers2grade.zip')
        except:
            pass

def start_grading():
    st.session_state['grading'] = True
    st.session_state['pts_possible'] = st.session_state.pts_possible_choice

def finish_grading(grid=None):
    # st.session_state['grading'] = False
    st.session_state['finished_grading'] = True
    try:
        if not grid.empty:
            print('setting grid df')
            st.session_state['grades_df'] = grid
    except:
        pass

def set_pts_possible():
    st.session_state['pts_possible'] = st.session_state.pts_possible_choice

def set_file():
    st.session_state['uploaded_file'] = st.session_state.uploaded_file_choice

def help_zip():
    st.session_state['zipped_files'] = st.session_state.zip_file_choice

def accept_grades(grid=None):
    finish_grading()
    try:
        if not grid.empty:
            print('setting grid df')
            st.session_state['grades_df'] = grid['data']
    except:
        pass

# other functions
# IMPORTANT: Cache the conversion to prevent computation on every rerun
@st.cache(allow_output_mutation=True)
def convert_df(df):
    csv = df.to_csv().encode('utf-8')
    return csv

# Containers
header_container = st.container()
points_container = st.container()
stats_container = st.container()
df_container = st.container()
footer_container = st.container()

# App

# Initial state
if st.session_state['uploaded_file'] == False and st.session_state['models_loaded'] == False:
    with st.spinner("Warming up the grading robots... Grab some coffee, this could take a minute or two."):
        setup_folders()

# Models ready, need uploaded file
if st.session_state['uploaded_file'] == False:
    with header_container:
        col1, col2 = st.columns([2, 3])
        with col1:
            try:
                st.image("/app/paper-grading-assistant/streamlit/images/984102_avatar_casual_male_man_person_icon.png")
            except:
                st.image("./images/984102_avatar_casual_male_man_person_icon.png")

        with col2:
            st.write('') # spacer
            st.write('') # spacer
            st.write('') # spacer
            st.write('') # spacer
            st.title("Hi! I'm Ted, your grading assistant.")
        st.header("Ready to take back your planning period?")
        st.subheader("Just a few quick things...")
        st.write("Your data is totally safe! Everything will be dumped as soon as you close this tab.")

        st.write("The essays I grade currently need to all be in .docx or .pdf formats.")
        st.write("They will ALSO all need to be in a folder that's zipped.")
        with st.expander("Click here for a file zipper >"):
            st.file_uploader('Choose papers to grade here...', key='zip_file_choice', accept_multiple_files=True, type=['.docx', '.pdf'], on_change=help_zip)
            if st.session_state.zip_file_choice:
                for file_ in st.session_state.zip_file_choice:
                    save_uploaded_file(file_)
                make_archive('papers2grade', './data/', '.')
                for file_ in st.session_state.zip_file_choice:
                    os.remove('./data/'+file_.name)
                with open('papers2grade.zip', 'rb') as dl_file:
                    st.download_button(
                        label="Download zipped papers",
                        data=dl_file,
                        file_name='zipped_papers.zip',
                        mime='application/zip')
        st.write("Once that's all done, just drop the zipped file below!")

        uploaded_file = st.file_uploader('Upload student papers here', type=['.zip'], key='uploaded_file_choice', on_change=set_file)
        st.write("") # spacer
        st.write("Psst! Not sure about all this? Try it out with some [sample papers](https://github.com/maxemileffort/paper-grading-assistant/raw/master/streamlit/sample_data/grading_assistant_test.zip), or some papers from last year.")

# User selects scale for grading
if (st.session_state['grading'] == False 
    and st.session_state['uploaded_file'] != False
    and st.session_state['finished_grading'] == False
    ):
    with points_container:
        pts_possible = st.text_input('How many points are possible for this assignment?', 
                                      value="100", 
                                      on_change=set_pts_possible, 
                                      key='pts_possible_choice')
        st.write("") # spacer
        st.button("Start grading!", on_click=start_grading)

# Results
if st.session_state['grading'] == True and st.session_state['finished_grading'] == False:
    with stats_container:
        uf = st.session_state['uploaded_file']
        pp = int(st.session_state['pts_possible'])
        st.write("") # spacer
        msg = st.success("Successfully uploaded papers")
        with st.spinner("Sorting papers to give to the robots..."):
            extracted_papers = grade_papers(uf, pp)
        msg.empty()
        st.markdown("## Click the button below to see your results!")
        st.button("See Results >", on_click=finish_grading, kwargs={'grid': extracted_papers})
        
# TODO instead of trying to download raw results, make a new pane where each 
# paper is clickable with comments and grades added in. 
# IE Sidebar, click name, render paper with comments and grade, editable. 
# Once teacher is done checking everything out, re-build csv so that 
# grades are updated, essay is dropped, and everything is easier to read

if st.session_state['finished_grading'] == True:
    st.markdown("## How to use this spreadsheet:")
    st.markdown("### The next table will have some data that you can use to speed up your grading process. Chaching!")
    st.markdown("If it looks a little small, hover over the table and a little icon appears on the top right side. Click it, and it will make the table bigger.")
    with st.expander("Click here for a legend to the columns on the table >"):
        st.markdown("* **File:** The name of the file graded. Helpful if students put their own names in the actual file name.")
        st.markdown("* **Word Count:** Approximate number of words in the essay, not including one-letter words.")
        st.markdown("* **Page Count:** Approximate number of pages in this paper, based on the idea that pages are about 250 words.")
        st.markdown("* **Final Score:** The score recommended for the paper based on the maximum score submitted.")
        st.markdown("* **Essay:** The submitted essay.")
    
    with st.form("grade_review_choice"):
        st.subheader("Review Grades:")
        grid = AgGrid(st.session_state['grades_df'], editable=True, fit_columns_on_grid_load=True)
        submitted = st.form_submit_button(
            label="Accept Grades >",
        )
        
    if submitted:
        st.dataframe(grid['data'])
        csv = convert_df(grid['data'])
        st.download_button(
            label="Download grades as spreadsheet",
            data=csv,
            on_click=reset_state,
            file_name='graded_papers.csv',
            mime='text/csv')

# reset button in the footer
if st.session_state['uploaded_file'] != False:
    with footer_container:
        st.button("< Start over", on_click=reset_state)
