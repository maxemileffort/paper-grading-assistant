import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
import string
import re

import pandas as pd
import numpy as np

from utils import *

# Containers
header_container = st.container()
points_container = st.container()
stats_container = st.container()

# State
states = ['grading', 'uploaded_file', 'pts_possible']
for state in states:
    if state not in st.session_state:
        st.session_state[state] = False

# Methods
def start_grading():
    st.session_state['grading'] = True
    st.session_state['pts_possible'] = st.session_state.pts_possible_choice

def set_pts_possible():
    st.session_state['pts_possible'] = st.session_state.pts_possible_choice

def set_file():
    st.session_state['uploaded_file'] = st.session_state.uploaded_file_choice


if st.session_state['uploaded_file'] == False:
    with header_container:

        # for example a logo or a image that looks like a website header
        # st.image('logo.png')

        # different levels of text you can include in your app
        st.title("Hi! I'm Skip, your grading assistant.")
        st.header("Ready to take back your planning period?")
        st.subheader("Just a few quick things...")
        st.write("Your data is totally safe! Everything will be dumped as soon as you close this tab.")

        st.write("The essays I grade currently need to all be in .docx formats.")
        st.write("They will ALSO all need to be in a folder that's zipped.")
        st.write("Once that's all done, just drop the zipped file below!")

        uploaded_file = st.file_uploader('Upload student papers here', type=['.zip'], key='uploaded_file_choice', on_change=set_file)
        st.write("") # spacer
        # TODO fix this link to lead somewhere that actually has sample papers
        st.write("Psst! Not sure about all this? Try it out with some [sample papers](https://www.google.com), or some papers from last year.")

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
        st.write("") # spacer
        st.subheader("Grading these papers...")
        extracted_papers = grade_papers(st.session_state['uploaded_file'], 
                                        int(st.session_state['pts_possible']))
        st.dataframe(extracted_papers)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(extracted_papers)
        st.download_button(
            label="Download grades as CSV",
            data=csv,
            file_name='graded_papers.csv',
            mime='text/csv')
