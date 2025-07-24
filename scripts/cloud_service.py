import os
from google.cloud import storage

# Set Google credentials (this is the correct method)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/sina.tvk.1997/AI-weather-predictor/authentication/service_account.json"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to a bucket."""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# File paths
source_files = [
    '/home/sina.tvk.1997/AI-weather-predictor/data/raw_data.csv',
    '/home/sina.tvk.1997/AI-weather-predictor/data/X.npy',
    '/home/sina.tvk.1997/AI-weather-predictor/data/y.npy'
]

destination_folder = [
    'raw_data/raw_data.csv',
    'features/X.npy',
    'target/y.npy'
]

# Upload each file
for i in range(len(source_files)):
    upload_blob(
        bucket_name="eastern-bedrock-464312-h1_bucket",
        source_file_name=source_files[i],
        destination_blob_name=destination_folder[i]
    )