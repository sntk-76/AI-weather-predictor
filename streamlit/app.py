import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from data_collector import data_downloader
from preprocessing import main as preprocess_pipeline
from openai import OpenAI
from dotenv import load_dotenv
from sheet_logger import log_to_google_sheet
from google.oauth2.service_account import Credentials

# --- 🔐 Load API Key ---
key = st.secrets["OPENAI_API_KEY"]
openai = OpenAI(api_key=key)

# --- 🔮 Build Prompt for LLM ---
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

# --- LLM Call ---
def llm(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Cached Model Loader ---
@st.cache_resource
def load_weather_model(path="weather_model.keras"):
    return load_model(path)

# --- App UI Starts ---
st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌤️", layout="wide")
st.title("🌤️ AI Weather Forecast Assistant")
st.markdown("**Enter a city name below to get a 7-day AI-powered weather forecast with a natural language summary.**")

city = st.text_input("📍 Enter city name:")

if st.button("🔍 Get Forecast"):
    try:
        st.toast("🔄 Collecting weather data...")
        raw_df = data_downloader(city)

        if raw_df.empty or len(raw_df) < 37:
            st.error(f"❌ Not enough recent data for {city}. Got only {len(raw_df)} days.")
        else:
            st.toast("⚙️ Preprocessing...")
            X_all, _ = preprocess_pipeline(raw_df)
            X_input = X_all[-1:].astype(np.float32)

            st.toast("📦 Loading model...")
            model = load_weather_model(path="streamlit/weather_model.keras")

            st.toast("🤖 Predicting next 7 days...")
            y_pred = model.predict(X_input)[0]

            # --- Format forecast ---
            targets = ['tavg', 'tmin', 'tmax', 'wspd', 'prcp', 'snow']
            forecast_df = pd.DataFrame(y_pred, columns=targets)

            forecast_df[['tavg', 'tmin', 'tmax', 'snow']] = forecast_df[['tavg', 'tmin', 'tmax', 'snow']].round(0).astype(int)
            forecast_df['Rain?'] = np.where(forecast_df['prcp'] >= 0.5, 'Yes', 'No')
            forecast_df['Rain_Prob (%)'] = (forecast_df['prcp'] * 100).clip(lower=0).astype(int)
            forecast_df['Snow?'] = np.where(forecast_df['snow'] >= 0.5, 'Yes', 'No')
            forecast_df['Snow_Prob (%)'] = (forecast_df['snow'] * 100).clip(lower=0).astype(int)

            forecast_df.drop(columns=['prcp', 'snow', 'wspd'], inplace=True)
            forecast_df.rename(columns={
                'tavg': 'Avg Temp (°C)',
                'tmin': 'Min Temp (°C)',
                'tmax': 'Max Temp (°C)'
            }, inplace=True)
            forecast_df.index = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=7)
            forecast_df.index.name = "📅 Date"

            # --- Display forecast ---
            st.subheader(f"📊 7-Day Forecast for {city}")
            st.dataframe(forecast_df, use_container_width=True)

            # --- AI Summary ---
            st.subheader("📝 AI-Generated Weekly Summary")
            with st.spinner("⏳ Generating AI summary... Please wait."):
                prompt = build_prompt(forecast_df, city)
                summary = llm(prompt)
            st.markdown(summary, unsafe_allow_html=True)


            # --- Log query ---
            log_to_google_sheet(city)
            st.toast("Logged search for future analysis.")
            st.success("✅ Done! Stay dry or enjoy the sunshine! 😎")

    except Exception as e:
        st.error(f"🚨 Something went wrong:\n\n{e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <center>
        Made with ❤️ by Sina • 
        <a href="https://github.com/sntk-76/AI-weather-predictor" target="_blank">View Project on GitHub</a>
    </center>
    """,
    unsafe_allow_html=True
)