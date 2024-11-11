import streamlit as st
import pandas as pd

def load_data():
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df)
        return df
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None