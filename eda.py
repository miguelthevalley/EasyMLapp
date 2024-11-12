import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io

def initial_eda(df):
    st.write("## Initial EDA")
    if st.checkbox("Show Initial Data Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    if st.checkbox("Show Initial Descriptive Statistics"):
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())

    if st.checkbox("Show Initial Correlation Matrix"):
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Pairplot"):
        st.write("### Pairplot")
        pairplot_fig = sns.pairplot(df, diag_kind='kde')
        st.pyplot(pairplot_fig)

def final_eda(df):
    st.write("## Final EDA After Transformations")

    if st.checkbox("Show Final Data Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    if st.checkbox("Show Final Descriptive Statistics"):
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())

    if st.checkbox("Show Final Correlation Matrix"):
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.checkbox('Show Pairplot'):
        st.write('### Pairplot')
        st.pyplot(fig)