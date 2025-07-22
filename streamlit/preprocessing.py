import pandas as pd
import numpy as np
import os

def add_time_features(df):

    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['weekday'] = df['time'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['dayofyear'] = df['time'].dt.dayofyear
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    
    return df


def handle_missing_value(data):
    new_order = [
        'city', 'year', 'month', 'day', 'dayofyear',
        'dayofyear_sin', 'dayofyear_cos', 'weekday', 'is_weekend',
        'tavg', 'tmin', 'tmax', 'wspd', 'prcp', 'snow',
        'wspd_missing'
    ]

    snowy_cities = [
        "Moscow", "Toronto", "Chicago", "Helsinki", "Oslo", "Stockholm", "Tallinn", "Montreal", "Halifax",
        "Winnipeg", "Fairbanks", "Yellowknife", "Barrow (Utqiaġvik)", "Tromsø", "Novosibirsk", "Irkutsk",
        "Anchorage", "Vladivostok", "Astana", "Bishkek"
    ]

    for col in ['tavg', 'tmin', 'tmax', 'wspd', 'prcp', 'snow']:
        print(f"{col}: {data[col].isna().sum()} missing before filling")

    for col in ['wspd', 'prcp', 'snow']:
        data[f'{col}_missing'] = data[col].isna().astype(int)

    data.drop(columns=['wdir', 'tsun', 'wpgt', 'pres'], inplace=True, errors='ignore')

    data['snow'] = data.groupby('city')['snow'].transform(
        lambda x: x.fillna(x.rolling(window=7, min_periods=1).median()) if x.name in snowy_cities else x.fillna(0.0)
    )
    data['snow'] = data['snow'].fillna(0.0)

    data['prcp'] = data.groupby('city')['prcp'].transform(
        lambda x: x.fillna(x.rolling(window=7, min_periods=1).median())
    )
    data['prcp'] = data['prcp'].fillna(0.0)

    for col in ['tavg', 'tmin', 'tmax', 'wspd']:
        data[col] = data.groupby('city')[col].transform(
            lambda x: x.interpolate(limit=5, limit_direction='both')
        )

    for col in ['tavg', 'tmin', 'tmax', 'wspd']:
        data[col] = data.groupby('city')[col].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )

    data = data.dropna().reset_index(drop=True)

    return data



def add_derived_weather_features(df):
    
    df['temp_range'] = df['tmax'] - df['tmin']   
    df['snow_ratio'] = df.apply(
        lambda row: row['snow'] / row['prcp'] if row['prcp'] > 0 else 0.0, axis=1
    )

    df['avg_temp_change_3d'] = df.groupby('city')['tavg'].transform(
        lambda x: x.diff().rolling(window=3, min_periods=1).mean()
    )

    def compute_wind_chill(t, v):
        if pd.isna(t) or pd.isna(v):
            return np.nan
        if t > 10 or v <= 4.8:
            return t
        return 13.12 + 0.6215*t - 11.37*(v**0.16) + 0.3965*t*(v**0.16)

    df['wind_chill_index'] = df.apply(lambda row: compute_wind_chill(row['tavg'], row['wspd']), axis=1)
    df['is_freezing'] = (df['tmin'] <= 0).astype(int)
    df['heavy_precip'] = (df['prcp'] >= 10.0).astype(int)
    df['strong_wind'] = (df['wspd'] >= 40.0).astype(int)
    df['heatwave'] = (df['tmax'] >= 35.0).astype(int)
    df['cold_wave'] = (df['tmin'] <= -10.0).astype(int)

    return df



def create_sequences(df, input_len=30, output_len=7, features=None, targets=None):

    """
    Converts a time series dataframe into a list of (input_sequence, target_sequence) pairs
    for supervised learning with sequential models (e.g., LSTM, Transformer).

    Parameters:
    - df: Preprocessed pandas DataFrame containing all cities' weather data
    - input_len: Number of past days used as input (default is 30)
    - output_len: Number of future days to predict (default is 7)
    - features: List of feature columns to include in the input (X)
    - targets: List of target columns to include in the output (y)

    Returns:
    - sequences: A list of tuples, each containing:
        (input_array: shape [input_len, num_features],
         target_array: shape [output_len, num_targets])
    """

    sequences = []


    for city, group in df.groupby('city'):

        group = group.sort_values('time').reset_index(drop=True)
        

        for i in range(len(group) - input_len - output_len + 1):

            input_slice = group.iloc[i:i+input_len][features].values
            

            target_slice = group.iloc[i+input_len:i+input_len+output_len][targets].values


            sequences.append((input_slice, target_slice))

    return sequences



def prepare_numpy_batches(sequences):
    
    """
    Converts a list of (input_sequence, target_sequence) pairs into NumPy arrays
    that can be fed into machine learning models.

    Parameters:
    - sequences: List of tuples, where each tuple is:
        (input_array: shape [input_len, num_features],
         target_array: shape [output_len, num_targets])

    Returns:
    - X: NumPy array of shape [num_samples, input_len, num_features]
    - y: NumPy array of shape [num_samples, output_len, num_targets]
    """


    X = np.array([s[0] for s in sequences])
    y = np.array([s[1] for s in sequences])

    return X, y



def main(data):


    data = add_time_features(data)
    data = handle_missing_value(data)
    data = add_derived_weather_features(data)

    features = [
        'tavg', 'tmin', 'tmax', 'wspd', 'prcp', 'snow',
        'dayofyear_sin', 'dayofyear_cos',
        'month', 'weekday', 'is_weekend',
        'temp_range', 'snow_ratio', 'avg_temp_change_3d',
        'wind_chill_index', 'is_freezing', 'heavy_precip',
        'strong_wind', 'heatwave', 'cold_wave'
    ]

    targets = ['tavg', 'tmin', 'tmax', 'wspd', 'prcp', 'snow']
    data = data.dropna().reset_index(drop=True)

    sequences = create_sequences(
        df=data,
        input_len=30,
        output_len=7,
        features=features,
        targets=targets
    )

    X, y = prepare_numpy_batches(sequences)

    return X, y