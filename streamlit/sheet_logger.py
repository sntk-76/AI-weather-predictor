import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import streamlit as st

def log_to_google_sheet(
    city,
    sheet_id="1OB_3v48cYCH7VaXd-JMGVg93q3ga1f3kMyKLIXvQmsA"
):
    
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_info(
        st.secrets["gspread"],
        scopes=scope
    )

    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet("Logs")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worksheet.append_row([now, city])