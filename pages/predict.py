import streamlit as st
import pandas as pd
import numpy as np
import io
from tensorflow.keras.models import load_model
import os
from utils.db_utils import fetch_file_content, update_file_in_db
from datetime import datetime
from tensorflow.keras.models import load_model
import asyncio
import nest_asyncio
from requests_html import AsyncHTMLSession, HTMLSession

model_path = "models/model.kerasCOLD2025MARCHSMOTnewformat.h5"

try:
    # Load the model
    model = load_model(model_path)
    st.success("Model loaded successfully!")
    st.write("Model Summary:")
    #model.summary(print_fn=lambda x: st.text(x))  # Print model summary in Streamlit
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Debug: Print the current working directory
#st.write("Current working directory:", os.getcwd())

# Define the path to the model
model_path = "models/model.kerasCOLD2025MARCHSMOTnewformat.h5"

# Check if the file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Load the model
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Apply nest_asyncio for compatibility with async functions in Streamlit
nest_asyncio.apply()

# Asynchronous scraping function
async def scrape_odds_async(url_prefix, race_num):
    async_session = AsyncHTMLSession()
    url = f"{url_prefix}{race_num}"
    response = await async_session.get(url)
    await response.html.arender(sleep=20)  # Increase rendering time for dynamic content
    odds_values = []

    divs = response.html.find('div[id^="odds_WIN_"]')  # Locate divs containing odds
    seen_ids = set()  # Track unique IDs to avoid duplicates
    for div in divs:
        div_id = div.attrs.get("id")
        if div_id in seen_ids:
            continue
        seen_ids.add(div_id)

        odds_element = div.find("a", first=True) or div.find("span", first=True)
        if odds_element:
            odds_text = odds_element.text.strip()
            if odds_text not in ["", "--", "SCR"]:  # Skip invalid odds
                odds_values.append(odds_text)
    return odds_values

# Synchronous scraping function (fallback)
def scrape_odds_sync(url_prefix, race_num):
    sync_session = HTMLSession()
    url = f"{url_prefix}{race_num}"
    response = sync_session.get(url)
    odds_values = []

    divs = response.html.find('div[id^="odds_WIN_"]')  # Locate divs containing odds
    seen_ids = set()
    for div in divs:
        div_id = div.attrs.get("id")
        if div_id in seen_ids:
            continue
        seen_ids.add(div_id)

        odds_element = div.find("a", first=True) or div.find("span", first=True)
        if odds_element:
            odds_text = odds_element.text.strip()
            if odds_text not in ["", "--", "SCR"]:
                odds_values.append(odds_text)
    return odds_values

# Function to calculate len_race based on RacingDate and RaceNo
def get_len_race(csv_file, racing_date):
    # Read the CSV file with utf-8 encoding
    df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
    #st.write("Shape of full DataFrame:", df.shape)  # Debugging
    #st.write("First few rows of DataFrame:", df.head())  # Debugging

    # Ensure the RacingDate column exists
    if 'RacingDate' not in df.columns:
        st.error("The 'RacingDate' column is missing from the file.")
        return []

    # Debug: Check unique values in the RacingDate column
    #st.write("Unique RacingDate values before processing:", df['RacingDate'].unique())

    # Normalize RacingDate column (explicitly specify dayfirst=True for dd/mm/yyyy format)
    df['RacingDate'] = pd.to_datetime(df['RacingDate'], errors='coerce', dayfirst=True).dt.strftime("%d/%m/%Y").str.strip()

    # Debug: Check unique values after processing
    #st.write("Unique RacingDate values after processing:", df['RacingDate'].unique())

    # Debug: Check exact matches
    matches = df['RacingDate'] == racing_date
    #st.write(f"Number of exact matches for {racing_date}: {matches.sum()}")
    #st.write("Matching rows:", df[matches])

    # If no matches, return an empty list
    if matches.sum() == 0:
        st.warning(f"No data found for the selected date: {racing_date}")
        return []

    # Filter the data for the selected racing date
    df_filtered = df[matches]

    # Compute len_race
    len_race = df_filtered.groupby('RaceNo').size().tolist()
    st.write("len_race:", len_race)  # Debugging
    return len_race

# Main function for the Prediction Page
def show_prediction_page():

    if st.button("⬅️ Back to Main Page"):
        st.session_state["current_page"] = "main"
        return

    st.title("Prediction Page")

    # Display the selected file
    file_name = st.session_state.get("selected_file_name")
    file_id = st.session_state.get("selected_file_id")
    if not file_name or not file_id:
        st.warning("No file selected. Please go to the Main Page.")
        return

    st.write(f"Selected File: `{file_name}`")
    file_content = fetch_file_content(file_id)
    if file_content:
        df = pd.read_csv(io.StringIO(file_content), low_memory=False)
        st.dataframe(df)
    else:
        st.error("Failed to load the file content from the database.")
        return

    # Step 1: Update Win Odds
    st.subheader("Step 1: Update Win Odds")
    selected_date = st.date_input("Select a Date")
    selected_race_number = st.selectbox("Select Race Number", range(1, 15))
    selected_race_course = st.radio("Select Race Course", ["ST", "HV"])

    st.write(f"**Selected Date**: {selected_date}")
    st.write(f"**Selected Race Number**: {selected_race_number}")
    st.write(f"**Selected Race Course**: {selected_race_course}")

    if st.button("Update Win Odds"):
        try:
            # Convert selected date to dd/mm/yyyy and yyyy-mm-dd formats
            race_date_ddmmyyyy = selected_date.strftime("%d/%m/%Y")
            race_date_iso = selected_date.strftime("%Y-%m-%d")

            # Construct URL prefix
            url_prefix = f"https://bet.hkjc.com/en/racing/wp/{race_date_iso}/{selected_race_course}/"

            # Scrape odds
            try:
                odds_values = asyncio.run(scrape_odds_async(url_prefix, selected_race_number))
            except Exception as e:
                st.warning(f"Async scraping failed: {e}. Trying synchronous scraping.")
                odds_values = scrape_odds_sync(url_prefix, selected_race_number)

            # Update the DataFrame
            mask = (df['RacingDate'] == race_date_ddmmyyyy) & (df['RaceNo'] == selected_race_number)
            num_matching_rows = mask.sum()

            # Adjust odds values length
            if len(odds_values) > num_matching_rows:
                odds_values = odds_values[:num_matching_rows]
            elif len(odds_values) < num_matching_rows:
                odds_values.extend(["100"] * (num_matching_rows - len(odds_values)))

            # Convert odds to float
            odds_values = [float(odds) if odds != "100" else 100.0 for odds in odds_values]
            if len(odds_values) == num_matching_rows:
                df.loc[mask, 'Win Odds'] = odds_values
                st.success("Win Odds updated successfully!")
                st.dataframe(df)  # Display updated DataFrame
            else:
                st.error("Error: Length mismatch between odds and matching rows.")
                return

            # Save the updated DataFrame back to the database
            updated_csv = io.StringIO()
            df.to_csv(updated_csv, index=False)
            update_file_in_db(file_id, updated_csv.getvalue())  # Correct function to update the file
            st.success("Updated file saved to the database successfully!")

        except Exception as e:
            st.error(f"Error updating Win Odds: {e}")

    # Step 2: Run the Model
    st.subheader("Step 2: Run the Model")
    st.write("To run the model to predict the result.")
    if st.button("Run the Model"):
        try:
            # Convert the selected date to the format used in the CSV (dd/mm/yyyy)
            racing_date = selected_date.strftime("%d/%m/%Y")
            st.write(f"Selected Racing Date: {racing_date}")  # Debugging

            # Calculate len_race
            len_race = get_len_race(io.StringIO(file_content), racing_date)

            # Check if len_race is empty
            if not len_race:
                st.error("No races found for the selected date. Please check the date or file content.")
                return

            #st.write("len_race:", len_race)  # Debugging

            # Prepare data for prediction
            dataframe_predict1 = pd.read_csv(io.StringIO(file_content), low_memory=False)
            #st.write("Shape of Input DataFrame:", dataframe_predict1.shape)
            #st.write(dataframe_predict1.head())

            dataframe_predict = dataframe_predict1.drop(
                [
                    "Pla.", "Horse No.", "Horse", "Jockey", "Trainer", "Act. Wt.",
                    "Declar. Horse Wt.", "RacingDate", "RaceNo", "Incident", "HorseID",
                    "Rtg.", "Class", "DoHorse_last2", "Win123", "Win123withColdHorse",
                    "RC", "Track", "ClassRank", "Gear", "Y_DoHorse", "Dr.",
                    "R35_Class_increase", "Y_MRD", "LBW", "Extracted_LBW",
                    "Running Position", "Finish Time", "Dist.", "distanceRange"
                ],
                axis=1
            )
            #st.write("Shape of Processed DataFrame:", dataframe_predict.shape)

            # Normalize specific columns
            if "Win Odds" in dataframe_predict.columns:
                dataframe_predict["Win Odds"] = (dataframe_predict["Win Odds"] - dataframe_predict["Win Odds"].mean()) / dataframe_predict["Win Odds"].std()
            if "R2_Trainer_RaceNo" in dataframe_predict.columns:
                dataframe_predict["R2_Trainer_RaceNo"] = (dataframe_predict["R2_Trainer_RaceNo"] - dataframe_predict["R2_Trainer_RaceNo"].mean()) / dataframe_predict["R2_Trainer_RaceNo"].std()
            #st.write("Processed DataFrame after Normalization:", dataframe_predict.head())

            race_index = 0
            table_index = 0

            df_betting = pd.DataFrame()
            df_betting["RaceNumber"] = 0
            df_betting["NoOfHorse"] = 0
            #df_betting["NoOfHorseTopHalf"] = 0
            df_betting["sample(Real Place)"] = 0
            df_betting["score(Prediction)"] = 0
            df_betting["WinOdd"] = 0
            #df_betting["TotalScoreTopHalf"] = 0
            df_betting["PercentageOnScore"] = 0

            # Iterate through races and make predictions
            for index, no_of_horse in enumerate(len_race):
                #st.write(f"Processing Race {index + 1} with {no_of_horse} horses.")
                df_predict = pd.DataFrame()
                df_predict["sample(Real Place)"] = 0
                df_predict["score(Prediction)"] = 0
                df_predict["WinOdd"] = 0
                race_no = index + 1

                for sample in range(no_of_horse):
                    samples_to_predict = np.array([dataframe_predict.loc[race_index].values.tolist()])
                    #st.write("Samples to Predict:", samples_to_predict)  # Debug
                    predictions = model.predict(samples_to_predict, verbose=0)
                    #st.write("Predictions:", predictions)  # Debug

                    df_predict.loc[sample, "sample(Real Place)"] = sample + 1
                    df_predict.loc[sample, "score(Prediction)"] = predictions[0][0]
                    df_predict.loc[sample, "WinOdd"] = dataframe_predict1.loc[race_index, "Win Odds"]
                    df_betting.loc[race_index, "RaceNumber"] = race_no

                    race_index += 1

                df_sort = df_predict.sort_values("score(Prediction)", ascending=False, ignore_index=True)

                NoOfHorse = no_of_horse
                NoOfHorseTopHalf = NoOfHorse
                TotalScoreTopHalf = df_sort["score(Prediction)"][:NoOfHorseTopHalf].sum()

                for index2 in range(NoOfHorseTopHalf):
                    df_betting.loc[table_index, "RaceNumber"] = race_no
                    df_betting.loc[table_index, "sample(Real Place)"] = df_sort.loc[index2, "sample(Real Place)"]
                    df_betting.loc[table_index, "score(Prediction)"] = df_sort.loc[index2, "score(Prediction)"]
                    df_betting.loc[table_index, "WinOdd"] = df_sort.loc[index2, "WinOdd"]
                    df_betting.loc[table_index, "NoOfHorse"] = NoOfHorse
                    #df_betting.loc[table_index, "NoOfHorseTopHalf"] = NoOfHorseTopHalf
                    #df_betting.loc[table_index, "TotalScoreTopHalf"] = TotalScoreTopHalf
                    df_betting.loc[table_index, "PercentageOnScore"] = (
                        (df_sort.loc[index2, "score(Prediction)"] / TotalScoreTopHalf) * 100
                    )

                    table_index += 1

            st.subheader("Prediction Results")
            st.dataframe(df_betting)

        except Exception as e:
            st.error(f"Error running the model: {e}")