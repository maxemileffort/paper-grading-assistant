import streamlit as st

import numpy as np
import pandas as pd

from utils import *

# Config
set_config()

# State
states = ['grading', 'uploaded_file', 'pts_possible', 'models_loaded']
for state in states:
    if state not in st.session_state:
        st.session_state[state] = False

# State managers
def reset_state():
    for state in states:
        st.session_state[state] = False

def start_grading():
    st.session_state['grading'] = True
    st.session_state['pts_possible'] = st.session_state.pts_possible_choice

def set_pts_possible():
    st.session_state['pts_possible'] = st.session_state.pts_possible_choice

def set_file():
    st.session_state['uploaded_file'] = st.session_state.uploaded_file_choice

# Containers
header_container = st.container()
points_container = st.container()
stats_container = st.container()
footer_container = st.container()

# App
if st.session_state['uploaded_file'] == False and st.session_state['models_loaded'] == False:
    with st.spinner("Warming up the grading robots... Grab some coffee, this could take a minute or two."):
        setup_folders()
        
if st.session_state['uploaded_file'] == False:
    with header_container:
        # different levels of text you can include in your app
        st.title("Hi! I'm Skip, your grading assistant.")
        st.header("Ready to take back your planning period?")
        st.subheader("Just a few quick things...")
        st.write("Your data is totally safe! Everything will be dumped as soon as you close this tab.")

        st.write("The essays I grade currently need to all be in .docx or .pdf formats.")
        st.write("They will ALSO all need to be in a folder that's zipped.")
        st.write("Once that's all done, just drop the zipped file below!")

        uploaded_file = st.file_uploader('Upload student papers here', type=['.zip'], key='uploaded_file_choice', on_change=set_file)
        st.write("") # spacer
        # TODO fix this link to lead somewhere that actually has sample papers
        st.write("Psst! Not sure about all this? Try it out with some [sample papers](https://github.com/maxemileffort/paper-grading-assistant/raw/master/streamlit/sample_data/grading_assistant_test.zip), or some papers from last year.")

if st.session_state['grading'] == False and st.session_state['uploaded_file'] != False:
    with points_container:
        pts_possible = st.text_input('How many points are possible for this assignment?', 
                                      value="100", 
                                      on_change=set_pts_possible, 
                                      key='pts_possible_choice')
        st.write("") # spacer
        st.button("Start grading!", on_click=start_grading)

if st.session_state['grading'] == True:
    with stats_container:
        uf = st.session_state['uploaded_file']
        pp = int(st.session_state['pts_possible'])
        st.write("") # spacer
        st.subheader("Grading these papers...")
        extracted_papers = grade_papers(uf, pp)
        st.markdown("## How to use this spreadsheet:")
        st.markdown("### The next table will have some data that you can use to speed up your grading process. Chaching!")
        st.markdown("If it looks a little small, hover over the table and a little icon appears on the top right side. Click it, and it will make the table bigger.")
        st.markdown("Here's a key for the columns on the table:")
        st.markdown("* **File:** The name of the file graded. Helpful if students put their own names in the actual file name.")
        st.markdown("* **Essay:** The submitted essay.")
        st.markdown("* **Word Count:** Approximate number of words in the essay, not including one-letter words.")
        st.markdown("* **Page Count:** Approximate number of pages in this paper. This column is based on the idea that pages are about 250 words when double-spaced.")
        st.markdown("* **Letter Grade:** The letter grade recommended for the paper.")
        st.markdown("* **Organization Score:** Possible feedback for the student, based on how well the paragraphs are organized.")
        st.dataframe(extracted_papers)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(extracted_papers)
        st.download_button(
            label="Download grades as CSV",
            data=csv,
            on_click=reset_state,
            file_name='graded_papers.csv',
            mime='text/csv')

if st.session_state['uploaded_file'] != False:
    with footer_container:
        st.button("< Start over", on_click=reset_state)
