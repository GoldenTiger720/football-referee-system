from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import time
import os
import requests
from datetime import datetime, timedelta
import base64

from moviepy.editor import VideoFileClip

def avi_to_mp4(input_path):
    try:
        # Determine the output path
        output_path = input_path.replace('.avi', '.mp4')

        # Load the AVI file
        video_clip = VideoFileClip(input_path)
        
        # Write the video clip to a MP4 file
        video_clip.write_videofile(output_path, codec='h264_nvenc', fps=23,ffmpeg_params=['-movflags', 'faststart'])
        #video_clip.write_videofile("output.mp4", codec='libx264', ffmpeg_params=['-movflags', 'faststart'], fps=video_clip.fps)
              
        # Close the video clip
        video_clip.close()
        
        print("Conversion successful!")
        return output_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def decode_and_extract_id(encoded_string):
    # Decode from Base64
    decoded_bytes = base64.b64decode(encoded_string)

    # Decode from bytes to string
    decoded_string = decoded_bytes.decode('utf-8')

    # Split the decoded string by ':' and return the second part
    return decoded_string.split(':')[1]

def upload_file_to_cloud(orig_file_path, base64_sportunity_id):
    file_path = avi_to_mp4(orig_file_path)
    return
    storage_acct = "sportunitydiag304"  # Hardcoded storage account
    storage_key = "zJk05uzQ/GHKAuV2jP62BweO5FYjEFzXicuX4PiRbuBMIv9SLnYBwLIhdzmzOHjVaeMQycO4XjFhUiSelcwRaA=="  # Hardcoded storage key
    container_name = 'eventvideos'
    file_name = os.path.basename(file_path)
    shared_access_policy = BlobSasPermissions(read=True)

    blob_service = BlobServiceClient(account_url=f"https://{storage_acct}.blob.core.windows.net", credential=storage_key)
    blob_client = blob_service.get_blob_client(container=container_name, blob=file_name)

    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)

    sas_token = generate_blob_sas(
        blob_service.account_name,
        container_name,
        file_name,
        account_key=storage_key,
        permission=shared_access_policy,
        start=datetime.utcnow(),
        expiry=datetime.utcnow() + timedelta(days=365)  # Valid for 1 year
    )
    file_url = f"{blob_client.url}?{sas_token}"
    print(file_url)

    send_video_url(file_url, base64_sportunity_id)

    # Delayed deletion of local file
    time.sleep(3)
    os.remove(file_path)
    print("File deleted")

def send_video_url(video_url, base64_sportunity_id):
    config_backend_url = "https://backendsportunity2017.com/devices-hooks"
    username = "Sportunity"
    password = "zI4pN5wA8uK9aR1b"
    sportunity_ids = decode_and_extract_id(base64_sportunity_id)
    url = f"{config_backend_url}/store-video-url?video_url={video_url}&sportunity_ids={sportunity_ids}"

    print("Call backend URL to save the video:", url)

    response = requests.get(url, auth=(username, password))
    print("Response:", response)

# Assuming this code runs under Python 3.6+
if __name__ == "__main__":
    upload_file_to_cloud('C:/Develop/VideoOutput/broadcast_63200187.avi', "U3BvcnR1bml0eTo2NjM5ZGJmZDUzYWQ5YjAwNDgyNTQ5ZTU=")
