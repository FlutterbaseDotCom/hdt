import os
import h5py
import numpy as np
import requests

from google.cloud import storage

# Function to save data to HDF5
def save_to_hdf5(data, filename, use_compression=False):
    with h5py.File(filename, 'w') as hdf5_file:
        for key, value in data.items():
            group = hdf5_file.create_group(key)
            for i, array in enumerate(value):
                if use_compression:
                    group.create_dataset(str(i), data=array, compression="gzip")
                else:
                    group.create_dataset(str(i), data=array)
    print(f"Data saved to {filename}")

# Function to load data from HDF5
def load_from_hdf5(filename):
    data = {}
    with h5py.File(filename, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            group = hdf5_file[key]
            data[key] = [np.array(group[str(i)]) for i in range(len(group))]
    return data

# Function to authenticate GCS if necessary
def authenticate_gcs():
    if not os.path.exists('/content/adc.json'):
        from google.colab import auth
        auth.authenticate_user()
    print("Authenticated to GCS")

# Function to upload file to GCS bucket
def upload_to_gcs( source_file_name, destination_blob_name, bucket_name = 'yakiv_dt', ):
    authenticate_gcs()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}")

# Function to download file from GCS bucket
def download_from_gcs( source_blob_name, destination_file_name, bucket_name = 'yakiv_dt'):
    authenticate_gcs()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}")

def download_large_file_from_gcs_public(source_blob_name, destination_file_name, bucket_name='yakiv_dt', chunk_size=128):
    # Construct the public URL for the blob
    public_url = f'https://storage.googleapis.com/{bucket_name}/{source_blob_name}'

    # Stream the download
    with requests.get(public_url, stream=True) as r:
        r.raise_for_status()
        with open(destination_file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size*1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}")
