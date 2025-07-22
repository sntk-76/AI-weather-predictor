import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from data_collector import data_downloader
from preprocessing import main as preprocess_pipeline
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv("/home/sina.tvk.1997/AI-weather-predictor/authentication/keys.env")
key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=key)

def ML_model_prediction(city) : 

    model_path = "/home/sina.tvk.1997/AI-weather-predictor/models/weather_model.keras"

    print(f"Downloading recent data for {city}...")
    raw_df = data_downloader(city)

    if raw_df.empty or len(raw_df) < 37:
        raise ValueError(f" Not enough data for prediction (need at least 30 days). Got {len(raw_df)} days.")


    print("Running preprocessing...")
    X_all, _ = preprocess_pipeline(raw_df)

 
    X_input = X_all[-1:]
    X_input = np.array(X_input).astype(np.float32)

 
    print("Loading trained model...")
    model = load_model(model_path)


    print("Predicting next 7 days...")
    y_pred = model.predict(X_input)[0]


    targets = ['tavg', 'tmin', 'tmax', 'wspd', 'prcp', 'snow']
    forecast_df = pd.DataFrame(y_pred, columns=targets)

    forecast_df[['tavg', 'tmin', 'tmax']] = forecast_df[['tavg', 'tmin', 'tmax']].round(0).astype(int)


    forecast_df['prcp_raw'] = forecast_df['prcp'].clip(0, 1)
    forecast_df['snow_raw'] = forecast_df['snow'].clip(0, 1)


    forecast_df['Rain?'] = np.where(forecast_df['prcp_raw'] >= 0.5, 'Yes', 'No')
    forecast_df['Rain_Prob (%)'] = (forecast_df['prcp_raw'] * 100).round(0).astype(int)

    forecast_df['Snow?'] = np.where(forecast_df['snow_raw'] >= 0.5, 'Yes', 'No')
    forecast_df['Snow_Prob (%)'] = (forecast_df['snow_raw'] * 100).round(0).astype(int)

    forecast_df.drop(columns=['wspd', 'prcp', 'snow', 'prcp_raw', 'snow_raw'], inplace=True)

    forecast_df.rename(columns={
        'tavg': 'Avg Temp (°C)',
        'tmin': 'Min Temp (°C)',
        'tmax': 'Max Temp (°C)'
    }, inplace=True)

    forecast_df.index = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=7)
    forecast_df.index.name = "Date"

    return forecast_df



def build_prompt(forecast_df,city):
    
    prompt = f"Provide a comprehensive weather summary and insights for the upcoming 7 days in {city}. Use the forecast below to describe temperature trends, rain/snow chances, and any interesting patterns or warnings.\n\n"

    prompt += "7-Day Forecast:\n"
    for idx, row in forecast_df.iterrows():
        date_str = idx.strftime("%A, %B %d")
        prompt += (
            f"{date_str}:\n"
            f" - Avg Temp: {row['Avg Temp (°C)']}°C\n"
            f" - Min Temp: {row['Min Temp (°C)']}°C\n"
            f" - Max Temp: {row['Max Temp (°C)']}°C\n"
            f" - Rain: {row['Rain?']} (Probability: {row['Rain_Prob (%)']}%)\n"
            f" - Snow: {row['Snow?']} (Probability: {row['Snow_Prob (%)']}%)\n"
            "\n"
        )

    prompt += (
        "Please analyze this forecast and generate a natural language weather summary for the city. "
        "Include practical suggestions (e.g., umbrella, outdoor plans), any warnings for severe weather, "
        "and note any interesting trends (cooling, warming, dry spell, etc.)."
    )

    return prompt



def llm(prompt, openai_client):
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content



def main(city):

    forecast_df = ML_model_prediction(city)
    
    prompt = build_prompt(forecast_df, city)

    summary = llm(prompt, openai_client)

    return summary