import streamlit as st
from utils.db_utils import save_file_to_db, get_all_files, fetch_file_content

def show_main_page():
    st.title("Main Page")

    # Upload a new CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_content = uploaded_file.getvalue().decode("utf-8")
        save_file_to_db(file_name, file_content)
        st.success(f"File `{file_name}` uploaded successfully!")

    # List of uploaded files
    st.subheader("Uploaded Files")
    files = get_all_files()
    if not files:
        st.info("No files uploaded yet.")
    else:
        for file in files:
            st.write(f"**{file['name']}**")

            # Preview button
            if st.button(f"Preview {file['name']}", key=f"preview_{file['id']}"):
                content = fetch_file_content(file["id"])
                st.write(content)

            # Download button with a unique key
            st.download_button(
                label=f"Download {file['name']}",
                data=fetch_file_content(file["id"]),
                file_name=file["name"],
                key=f"download_{file['id']}"  # Unique key for each download button
            )

            # Proceed to Predict button with a unique key
            if st.button(f"Proceed to Predict {file['name']}", key=f"predict_{file['id']}"):
                st.session_state["selected_file_id"] = file["id"]
                st.session_state["selected_file_name"] = file["name"]
                st.session_state["current_page"] = "predict"