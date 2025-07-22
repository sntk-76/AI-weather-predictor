import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def log_to_google_sheet(city, sheet_id="1OB_3v48cYCH7VaXd-JMGVg93q3ga1f3kMyKLIXvQmsA", creds_path="/home/sina.tvk.1997/AI-weather-predictor/authentication/service_account.json"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet("Logs")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worksheet.append_row([now, city])