from meteostat import Daily, Point, Stations
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import pandas as pd

def data_downloader(city):

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=40)


    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    try:

        geolocator = Nominatim(user_agent="AI-Weather-Predictor")
        location = geolocator.geocode(city, timeout=10)

        if not location:
            print(f"Could not geocode {city}")
            return pd.DataFrame()

        lat, lon = location.latitude, location.longitude


        stations = Stations().nearby(lat, lon).fetch(25)
        if stations.empty:
            print(f"⚠️ No stations found near {city}")
            return pd.DataFrame()


        stations['coverage_days'] = (stations['daily_end'] - stations['daily_start']).dt.days
        stations = stations.sort_values(by='coverage_days', ascending=False)


        for station_id in stations.index:
            try:
                daily_data = Daily(station_id, start=start_date, end=end_date).fetch()
                if not daily_data.empty:
                    daily_data["city"] = city
                    return daily_data.reset_index()
            except Exception as inner_e:
                print(f"Failed to fetch data from station {station_id}: {inner_e}")

        print(f"None of the nearby stations have usable data for {city} in the last 14 days")
        return pd.DataFrame()

    except Exception as e:
        print(f"Error for {city}: {e}")
        return pd.DataFrame()