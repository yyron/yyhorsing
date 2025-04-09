import streamlit as st

def show_login_page():
    st.title("Login Page")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if username == "yyron" and password == "yyhh1234":
            st.session_state["logged_in"] = True
            st.session_state["current_page"] = "main"
        else:
            st.error("Invalid username or password!")