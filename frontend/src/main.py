import streamlit as st
try:
    st.title('Coffee & Health Analysis')
    st.write('Welcome to the dashboard!')
except Exception as e:
    st.error(f"An error occurred: {e}")