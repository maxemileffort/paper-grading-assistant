import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
import string
import re

import pandas as pd
import numpy as np

from utils import *

header_container = st.container()
stats_container = st.container()

uploaded_file = None

if uploaded_file == None:
    with header_container:

        # for example a logo or a image that looks like a website header
        # st.image('logo.png')

        # different levels of text you can include in your app
        st.title("Hi! I'm Jordie, your grading assistant.")
        st.header("Ready to take back your planning period?")
        st.subheader("Just a few quick things...")
        st.write("Your data is totally safe! Everything will be dumped as soon as you close this tab.")

        st.write("The essays I grade currently need to all be in .docx formats.")
        st.write("They will ALSO all need to be in a folder that's zipped.")
        st.write("Once that's all done, just drop the zipped file below!")

        uploaded_file = st.file_uploader('Upload student papers here', type=['.zip'])
        st.write("") # spacer
        # TODO fix this link to lead somewhere that actually has sample papers
        st.write("Psst! Not sure about all this? Try it out with some [sample papers](https://www.google.com), or some papers from last year.")

if uploaded_file !=  None:
    with stats_container:
        st.subheader("Grading these papers...")
        extracted_papers = grade_papers(uploaded_file)
        st.dataframe(extracted_papers)

# @st.cache
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')
# csv = convert_df(my_large_df)
# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='large_df.csv',
#     mime='text/csv')
