import streamlit as st
from pages.login import show_login_page
from pages.main import show_main_page
from pages.predict import show_prediction_page
from utils.db_utils import init_db

# Initialize the database
init_db()

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "login"
if "selected_file_id" not in st.session_state:
    st.session_state["selected_file_id"] = None
if "selected_file_name" not in st.session_state:
    st.session_state["selected_file_name"] = None

# Define navigation logic
def navigate_to_login():
    st.session_state["current_page"] = "login"
    st.session_state["logged_in"] = False

def navigate_to_main():
    st.session_state["current_page"] = "main"

def navigate_to_predict():
    st.session_state["current_page"] = "predict"

# Render the appropriate page
if st.session_state["current_page"] == "login":
    show_login_page()
elif st.session_state["current_page"] == "main":
    if st.session_state["logged_in"]:
        show_main_page()
    else:
        st.warning("You need to log in first!")
        navigate_to_login()
elif st.session_state["current_page"] == "predict":
    if st.session_state["logged_in"]:
        show_prediction_page()
    else:
        st.warning("You need to log in first!")
        navigate_to_login()