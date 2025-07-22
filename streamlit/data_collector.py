from meteostat import Daily, Stations
from datetime import datetime, timedelta
import pandas as pd
from opencage.geocoder import OpenCageGeocode
import streamlit as st

def data_downloader(city):
    # --- Set time range ---
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=40)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    try:
        # --- Use OpenCage Geocoder ---
        api_key = st.secrets["OPENCAGE_API_KEY"]
        geocoder = OpenCageGeocode(api_key)
        results = geocoder.geocode(city)

        if not results:
            print(f"❌ Could not geocode {city}")
            return pd.DataFrame()

        lat = results[0]['geometry']['lat']
        lon = results[0]['geometry']['lng']

        # --- Get weather stations near coordinates ---
        stations = Stations().nearby(lat, lon).fetch(25)
        if stations.empty:
            print(f"⚠️ No stations found near {city}")
            return pd.DataFrame()

        stations['coverage_days'] = (stations['daily_end'] - stations['daily_start']).dt.days
        stations = stations.sort_values(by='coverage_days', ascending=False)

        # --- Try downloading from the best station ---
        for station_id in stations.index:
            try:
                daily_data = Daily(station_id, start=start_date, end=end_date).fetch()
                if not daily_data.empty:
                    daily_data["city"] = city
                    return daily_data.reset_index()
            except Exception as inner_e:
                print(f"⚠️ Failed to fetch data from station {station_id}: {inner_e}")

        print(f"⚠️ No usable station data found for {city}")
        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error for {city}: {e}")
        return pd.DataFrame()
