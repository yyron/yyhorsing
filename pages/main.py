import streamlit as st
from utils.db_utils import save_file_to_db, get_all_files, fetch_file_content, delete_file_from_db

def show_main_page():
    st.title("Main Page")

    # Use session state to track uploads
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = set()  # Track successfully uploaded files

    # Upload a new CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_content = uploaded_file.getvalue().decode("utf-8")

        # Check if the file has already been uploaded in this session
        if file_name in st.session_state["uploaded_files"]:
            st.warning(f"The file `{file_name}` has already been uploaded in this session.")
        else:
            try:
                save_file_to_db(file_name, file_content)
                st.session_state["uploaded_files"].add(file_name)  # Mark as uploaded
                st.success(f"File `{file_name}` uploaded successfully!")
                st.experimental_rerun()  # Refresh the page to reflect the new file in the list
            except ValueError as e:
                st.error(str(e))

    # List of uploaded files
    st.subheader("Uploaded Files")
    files = get_all_files()
    if not files:
        st.info("No files uploaded yet.")
    else:
        for file in files:
            col1, col2 = st.columns([4, 1])  # Adjust column widths if necessary
            with col1:
                st.write(f"**{file['name']}**")
            with col2:
                if st.button(f"❌ Remove", key=f"remove_{file['id']}"):
                    delete_file_from_db(file["id"])
                    st.success(f"File `{file['name']}` removed successfully!")
                    st.experimental_rerun()  # Refresh the page

            # Action buttons below the file
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(f"Preview {file['name']}", key=f"preview_{file['id']}"):
                    content = fetch_file_content(file["id"])
                    st.write(content)
            with col2:
                st.download_button(
                    label=f"Download {file['name']}",
                    data=fetch_file_content(file["id"]),
                    file_name=file["name"],
                    key=f"download_{file['id']}"
                )
            with col3:
                if st.button(f"Proceed to Predict {file['name']}", key=f"predict_{file['id']}"):
                    st.session_state["selected_file_id"] = file["id"]
                    st.session_state["selected_file_name"] = file["name"]
                    st.session_state["current_page"] = "predict"