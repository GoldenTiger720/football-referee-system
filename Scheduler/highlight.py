import pychromecast
import time
import json
import os
SERVER_CONFIG = "c:/Develop/Configuration/server_config.json"

import shutil

import psutil

def get_file_size(file_path):
    """
    Get the size of a file in bytes.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        int: The size of the file in bytes.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No file found at: {file_path}")
    
    file_size = os.path.getsize(file_path)
    return file_size

def is_file_closed(file_path):
    for proc in psutil.process_iter(['open_files']):
        try:
            if any(file.path == file_path for file in proc.info['open_files'] or []):
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return True

def copy_file(source_folder, destination_folder, filename):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)

    # Check if the source file exists
    if not os.path.isfile(source_path):
        print(f"File {filename} does not exist in the source folder.")
        return

    # Create the destination folder if it does not exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy the file
    try:
        shutil.copy2(source_path, destination_path)
        print(f"File {filename} copied from {source_folder} to {destination_folder}.")
    except Exception as e:
        print(f"Error copying file {filename}: {e}")

def list_available_chromecasts():
    # Discover all Chromecast devices
    chromecasts, browser = pychromecast.get_chromecasts()
    if not chromecasts:
        print("No Chromecast devices found.")
        return

    print("Available Chromecast devices:")
    for cast in chromecasts:
        print(f" - {cast.name}")

def cast_video_to_chromecast(chromecast_name, video_file, timeout=20):
    chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=[chromecast_name])
    print("discovering device")
    if not chromecasts:
        print(f"No Chromecast with name '{chromecast_name}' found.")
        return

    # Connect to the Chromecast
    print("connecting to device")
    cast = chromecasts[0]
    cast.wait()

    # Get the media controllerd
    mc = cast.media_controller

    # Load and play the video file
    mc.play_media(video_file, 'video/mp4')
    mc.block_until_active()

    print(f"Playing '{video_file}' on '{chromecast_name}'")

    # Keep the script running while the video is playing
    ccntr=15
    while mc.status.player_state != 'IDLE' or ccntr>0:
        print("mc.status.player_state", mc.status.player_state)
        time.sleep(1)
        ccntr-=1

    print("Playback finished")

def load_config(filename):
    """Load JSON configuration from a file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        return None

if __name__ == "__main__":
    import sys


    server_config = load_config(SERVER_CONFIG)
    if not server_config:
        sys.exit(1)


    ip = server_config['server_ip']
    port = server_config['http_port']
    folder_path = server_config['highlight_folder']
    save_path = server_config['highlight_save']
    chromecast_name = server_config['chromecast']
    url = server_config['url']
    

    while True:
        current_files = set(os.listdir(folder_path))

        for file_name in current_files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                print(f"New file detected: {file_name}")
                try:
                    video_file = url+file_name
                    print("video to be played:", video_file)

                    last_file_size=-1
                    cntr=25
                    same_size_cntr=0
                    while (cntr>0):
                        cntr-=1
                        if (get_file_size(file_path)==last_file_size):
                            same_size_cntr+=1
                            if same_size_cntr>3:
                                break
                        last_file_size=get_file_size(file_path)
                        print("Video is still being written. waiting...", get_file_size(file_path))
                        time.sleep(1)


                    copy_file(folder_path, save_path, file_name)
                    cast_video_to_chromecast(chromecast_name, video_file)
                    time.sleep(15)
                    
                    os.remove(file_path)
                    
                    print(f"File {file_name} deleted.")
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")

        time.sleep(5)

    filename=""
    video_file = f'{url}/{filename}'
    cast_video_to_chromecast(chromecast_name, video_file)
    