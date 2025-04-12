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
from playwright.async_api import async_playwright
from requests_html import AsyncHTMLSession, HTMLSession



nest_asyncio.apply()

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

# Asynchronous odds scraping function using Playwright
async def scrape_odds_with_playwright(url_prefix, race_num):
    """
    Scrape odds data for a specific race using Playwright.
    """
    url = f"{url_prefix}{race_num}"
    odds_values = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)

            # Locate and extract odds data
            divs = await page.query_selector_all('div[id^="odds_WIN_"]')  # Find divs with odds data
            seen_ids = set()  # Track unique IDs to avoid duplicates

            for div in divs:
                div_id = await div.get_attribute("id")
                if div_id in seen_ids:
                    continue
                seen_ids.add(div_id)

                odds_element = await div.query_selector("a") or await div.query_selector("span")
                if odds_element:
                    odds_text = await odds_element.inner_text()
                    odds_text = odds_text.strip()
                    if odds_text not in ["", "--", "SCR"]:  # Skip invalid odds
                        odds_values.append(odds_text)

            await browser.close()
    except Exception as e:
        st.warning(f"Playwright async scraping failed: {e}")
        raise e  # Propagate the exception to trigger fallback

    return odds_values

# Synchronous odds scraping function using requests-html
def scrape_odds_sync(url_prefix, race_num):
    """
    Synchronous scraping using requests-html (HTMLSession).
    """
    sync_session = HTMLSession()
    url = f"{url_prefix}{race_num}"
    odds_values = []

    try:
        response = sync_session.get(url)
        divs = response.html.find('div[id^="odds_WIN_"]')
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
    except Exception as e:
        st.warning(f"Synchronous scraping failed: {e}")
        raise e  # Propagate the exception if sync scraping also fails

    return odds_values



# Function to scrape Pla. data using Playwright
async def scrape_pla_with_playwright(race_date, race_course, race_no):
    """
    Scrape Pla. and Horse No. for a specific race using Playwright.
    """
    url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={race_date}&Racecourse={race_course}&RaceNo={race_no}"
    pla_horse_data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Check if results are available
        page_content = await page.content()
        if "Information will be released shortly." in page_content:
            return None

        # Extract placing and horse numbers
        rows = await page.query_selector_all("table tbody tr")
        for row in rows:
            cols = await row.query_selector_all("td")
            if len(cols) >= 3:
                pla = await cols[0].inner_text()
                horse_no = await cols[1].inner_text()
                if pla.isdigit() and horse_no.isdigit():
                    pla_horse_data.append((int(pla.strip()), int(horse_no.strip())))

        await browser.close()
    return pla_horse_data

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

# Function to update Win Odds
def update_win_odds(df, file_id, selected_date, selected_race_number, selected_race_course):
    """
    Function to scrape and update Win Odds in the DataFrame.
    Tries Playwright-based async scraping first, falls back to requests-html sync scraping if needed.
    """
    try:
        # Convert selected date to dd/mm/yyyy and yyyy-mm-dd formats
        race_date_ddmmyyyy = selected_date.strftime("%d/%m/%Y")
        race_date_iso = selected_date.strftime("%Y-%m-%d")

        # Construct URL prefix
        url_prefix = f"https://bet.hkjc.com/en/racing/wp/{race_date_iso}/{selected_race_course}/"

        # Scrape odds
        try:
            # Instead of asyncio.run, use an event loop compatible with Streamlit
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine in the running loop
                odds_values = loop.run_until_complete(scrape_odds_with_playwright(url_prefix, selected_race_number))
            else:
                # Create a new event loop if none is running
                odds_values = asyncio.run(scrape_odds_with_playwright(url_prefix, selected_race_number))
        except Exception as async_error:
            st.warning(f"Async scraping with Playwright failed: {async_error}. Switching to synchronous scraping.")
            try:
                # Fall back to sync scraping with requests-html
                odds_values = scrape_odds_sync(url_prefix, selected_race_number)
            except Exception as sync_error:
                st.error(f"Synchronous scraping also failed: {sync_error}")
                return

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
        df.loc[mask, 'Win Odds'] = odds_values

        # Save the updated DataFrame back to the database
        updated_csv = io.StringIO()
        df.to_csv(updated_csv, index=False)
        update_file_in_db(file_id, updated_csv.getvalue())  # Save changes to the same file in the database
        st.success("Win Odds updated successfully!")
        st.dataframe(df)  # Display updated DataFrame
    except Exception as e:
        st.error(f"Error updating Win Odds: {e}")

# Function to scrape and update Pla. data
async def scrape_and_update_pla(df, file_id, input_date, race_courses):
    race_date = datetime.strptime(input_date, "%d/%m/%Y").strftime("%Y/%m/%d")
    filtered_df = df[df["RacingDate"] == input_date]
    if filtered_df.empty:
        st.warning(f"No data found for the selected date: {input_date}")
        return

    largest_race_no = filtered_df["RaceNo"].max()

    for race_no in range(1, largest_race_no + 1):
        for race_course in race_courses:
            pla_horse_data = await scrape_pla_with_playwright(race_date, race_course, race_no)
            if pla_horse_data is None:
                continue

            # Update the DataFrame
            for pla, horse_no in pla_horse_data:
                mask = (df["RacingDate"] == input_date) & (df["RaceNo"] == race_no) & (df["Horse No."] == horse_no)
                df.loc[mask, "Pla."] = pla
            break

    # Save updated DataFrame
    updated_csv = io.StringIO()
    df.to_csv(updated_csv, index=False)
    update_file_in_db(file_id, updated_csv.getvalue())
    st.success("Pla. updated successfully!")

def update_all_rules(df, selected_date, file_id):
    """
    Function to calculate and update all rule-based columns in the DataFrame.
    Updates the database directly instead of saving to CSV files.
    """
    try:
        # Ensure selected_date is in the correct format
        racing_date = selected_date.strftime("%d/%m/%Y")

        # Rule 4: Trainer has a horse that raced but didn't place
        df["R4_Trainer_NoWin123B4"] = 0
        for i in range(len(df)):
            try:
                if df.loc[i, "RaceNo"] == 1:
                    df.loc[i, "R4_Trainer_NoWin123B4"] = 0
                else:
                    index_start = df.loc[df["RacingDate"] == df.loc[i, "RacingDate"]].index[0]
                    for ii in range(index_start, i):
                        if (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (float(df.loc[ii, "Pla."]) > 3):
                            df.loc[i, "R4_Trainer_NoWin123B4"] = 1
                            break
            except:
                pass

        # Rule 6: Trainer has a "killed hot" horse (Win Odds <= 5)
        df["R6_Trainer_KillHotB4"] = 0
        for i in range(len(df)):
            try:
                if df.loc[i, "RaceNo"] == 1:
                    df.loc[i, "R6_Trainer_KillHotB4"] = 0
                else:
                    index_start = df.loc[df["RacingDate"] == df.loc[i, "RacingDate"]].index[0]
                    for ii in reversed(range(index_start, i)):
                        if (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (float(df.loc[ii, "Pla."]) > 3) and (float(df.loc[ii, "Win Odds"]) <= 5):
                            df.loc[i, "R6_Trainer_KillHotB4"] = 1
                            break
                        elif (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (float(df.loc[ii, "Pla."]) <= 3):
                            df.loc[i, "R6_Trainer_KillHotB4"] = 0
                            break
            except:
                pass

        # Rule 10: Trainer and Jockey combination with no previous placements
        df["R10_same_TrainerJockey_NoWin123B4"] = 0
        for i in range(len(df)):
            try:
                if df.loc[i, "RaceNo"] == 1:
                    df.loc[i, "R10_same_TrainerJockey_NoWin123B4"] = 0
                else:
                    index_start = df.loc[df["RacingDate"] == df.loc[i, "RacingDate"]].index[0]
                    for ii in range(index_start, i):
                        if (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (df.loc[ii, "Jockey"] == df.loc[i, "Jockey"]) and (float(df.loc[ii, "Pla."]) > 3):
                            df.loc[i, "R10_same_TrainerJockey_NoWin123B4"] = 1
                            break
            except:
                pass

        # Rule 12: Trainer and Jockey combination with a "killed hot" horse (Win Odds <= 4)
        df["R12_same_TrainerJockey_KillHotB4"] = 0
        for i in range(len(df)):
            try:
                if df.loc[i, "RaceNo"] == 1:
                    df.loc[i, "R12_same_TrainerJockey_KillHotB4"] = 0
                else:
                    index_start = df.loc[df["RacingDate"] == df.loc[i, "RacingDate"]].index[0]
                    for ii in range(index_start, i):
                        if (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (df.loc[ii, "Jockey"] == df.loc[i, "Jockey"]) and (float(df.loc[ii, "Pla."]) > 3) and (float(df.loc[ii, "Win Odds"]) <= 4):
                            df.loc[i, "R12_same_TrainerJockey_KillHotB4"] = 1
                            break
            except:
                pass

        # Rule 37: Current horse is "hot" (Win Odds <= 6)
        df["R37_isHot"] = 0
        for i in range(len(df)):
            try:
                if float(df.loc[i, "Win Odds"]) <= 6:
                    df.loc[i, "R37_isHot"] = 1
            except:
                pass

        # Rule 38: Trainer hasn't won with a "cold" horse (Win Odds >= 12)
        df["R38_Trainer_notWinColdB4"] = 0
        for i in range(len(df)):
            try:
                if df.loc[i, "RaceNo"] == 1:
                    df.loc[i, "R38_Trainer_notWinColdB4"] = 1
                else:
                    index_start = df.loc[df["RacingDate"] == df.loc[i, "RacingDate"]].index[0]
                    for ii in reversed(range(index_start, i)):
                        if (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (float(df.loc[ii, "Pla."]) <= 3) and (float(df.loc[ii, "Win Odds"]) >= 12):
                            df.loc[i, "R38_Trainer_notWinColdB4"] = 0
                            break
            except:
                pass

        # Rule 50: Trainer has horses that raced but didn't place
        df["R50_Trainer_notWinB4"] = 0
        for i in range(len(df)):
            try:
                if df.loc[i, "RaceNo"] == 1:
                    df.loc[i, "R50_Trainer_notWinB4"] = 0
                else:
                    index_start = df.loc[df["RacingDate"] == df.loc[i, "RacingDate"]].index[0]
                    for ii in reversed(range(index_start, i)):
                        if (df.loc[ii, "Trainer"] == df.loc[i, "Trainer"]) and (float(df.loc[ii, "Pla."]) <= 3):
                            df.loc[i, "R50_Trainer_notWinB4"] = 0
                            break
            except:
                pass

        # Save updated DataFrame back to the database
        updated_csv = io.StringIO()
        df.to_csv(updated_csv, index=False)
        update_file_in_db(file_id, updated_csv.getvalue())

        st.success("All rules updated successfully!")
    except Exception as e:
        st.error(f"Error updating rules: {e}")

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

    # Step 1a: Update Win Odds
    st.subheader("Step 1: Update Win Odds")
    selected_date = st.date_input("Select a Date")
    selected_race_number = st.selectbox("Select Race Number", range(1, 15))
    selected_race_course = st.radio("Select Race Course", ["ST", "HV"])

    st.write(f"**Selected Date**: {selected_date}")
    st.write(f"**Selected Race Number**: {selected_race_number}")
    st.write(f"**Selected Race Course**: {selected_race_course}")

    if st.button("Update Win Odds"):
        update_win_odds(df, file_id, selected_date, selected_race_number, selected_race_course)

    # Step 1b: Add a separate button for updating Hot/Cold flags
    if st.button("Update Hot/Cold Flags"):
        try:
            # Update the "R37_isHot" flag based on the latest "Win Odds"
            st.subheader("Step 2: Update Hot/Cold Flags")
            for i in df.index:
                try:
                    # Reset the flag to 0 by default
                    df.loc[i, "R37_isHot"] = 0

                    # Set the flag to 1 if the Win Odds are <= 6
                    if float(df.loc[i, "Win Odds"]) <= 6:
                        df.loc[i, "R37_isHot"] = 1
                except:
                    pass  # Skip rows with invalid data

            # Save the updated DataFrame back to the database
            updated_csv = io.StringIO()
            df.to_csv(updated_csv, index=False)
            update_file_in_db(file_id, updated_csv.getvalue())  # Save changes to the same file in the database
            st.success("Hot/Cold flags updated successfully!")
            st.dataframe(df)  # Display updated DataFrame

        except Exception as e:
            st.error(f"Error updating Hot/Cold flags: {e}")

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

    # Step 3: Post Racing - Update Win Odds
    st.subheader("Step 3: Post Racing Updates")
    st.write("Update Win Odds after the race.")
    if st.button("Update Win Odds (Post Racing)"):
        update_win_odds(df, file_id, selected_date, selected_race_number, selected_race_course)
    # Button to update Pla.
    if st.button("Update Pla."):
        try:
            # Schedule the coroutine using asyncio.create_task
            loop = asyncio.get_event_loop()
            task = loop.create_task(scrape_and_update_pla(df, file_id, selected_date.strftime("%d/%m/%Y"), [selected_race_course]))
            st.info("Updating Pla. data... Please wait.")

            # Optionally, wait for the task to complete (blocks UI)
            loop.run_until_complete(task)

            st.success("Pla. data updated successfully!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error updating Pla.: {e}")
    # Button to update all rules
    if st.button("Update All Value in Rules"):
        update_all_rules(df, selected_date, file_id)