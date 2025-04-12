import tkinter as tk
from tkinter import messagebox
import time
import sys
import requests
from datetime import datetime, timedelta, timezone
from operator import itemgetter
import json
import subprocess
import psutil
import threading
from bs4 import BeautifulSoup

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import os
import requests
import base64
from moviepy.editor import VideoFileClip
from pymongo import MongoClient
from bson.objectid import ObjectId

TIME_OFFSET = 2 # 2 hours
STARTUP_DELAY = 1 #recording start x minutes after the start time

GRAPHQL_URL = 'https://backendsportunity2017.com/graphql'

CONFIG = "c:/Develop/Configuration/feed_config.json"
SERVER_CONFIG = "c:/Develop/Configuration/server_config.json"
scheduler_file = r"C:\Develop\Configuration\scheduler.json"
feed_batch_file = "C:/Develop/Configuration/start_feed.bat"
ai_batch_file = r"C:\Develop\Configuration\start_ai.bat"
feed_process_name = "CameraFeeds.exe"
ai_process_name = "python.exe"
ai_process_args = "main3.py"
FIELD_NAME = None

startup_time = datetime.now(timezone.utc) + timedelta(hours=TIME_OFFSET) + timedelta(seconds=5)


log_file = open('output.log', 'a')

def print_and_log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file, flush=True)

def add_video_highlight(base64_video_id, cntr, caption, highlight_time):
    try:
        print_and_log("add_video_highlight()")
        url = 'https://backendsportunity2017.com/graphql'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # Revised mutation to match the new requirements with variable definitions
        mutation = """
        mutation AddVideoHighlight($id: String!, $idx: String!, $event_time: Int!, $caption: String!) {
            newVARHighligh(input: {
                videoId: $id, 
                videoHighlight: {
                    index: $idx, 
                    time: $event_time, 
                    caption: $caption
                }
            }) {
                clientMutationId
            }
        }
        """
        
        # Variables to be sent with the mutation
        variables = {
            'id': base64_video_id,
            'idx': str(cntr),
            'event_time': int(highlight_time),
            'caption': caption
        }
        
        # Prepare the JSON payload for the mutation
        payload = {
            'query': mutation,
            'variables': variables
        }
        
        # Send the mutation request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check the response status
        if response.status_code == 200:
            print("Mutation successful. Response:", response.json())
        else:
            print("Failed to execute mutation. Status Code:", response.status_code, "Response:", response.json())
    except:
        print("Error while uploading highlights....")

def update_sportunity_scores(sportunity_id, score1, score2):
    try:   
        url = 'https://backendsportunity2017.com/graphql'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # Revised mutation to match the new requirements
        mutation = """
        mutation UpdateSportunityScores($id: String!, $score1: Int!, $score2: Int!) {
            updateSportunityScores(input: {
                sportunityId: $id,
                score1: $score1,
                score2: $score2
            }) {
                clientMutationId
            }
        }
        """
        
        # Variables to be sent with the mutation
        variables = {
            'id': sportunity_id,
            'score1': score1,
            'score2': score2
        }
        
        # Send the mutation request
        response = requests.post(url, json={'query': mutation, 'variables': variables}, headers=headers)
        
        # Check the response status
        if response.status_code == 200:
            print("Mutation successful. Response:", response.json())
        else:
            print("Failed to execute mutation. Status Code:", response.status_code, "Response:", response.json())
    except:
        print("Error while uploading scores....")

'''def avi_to_mp4(input_path):
    try:
        # Determine the output path
        output_path = input_path.replace('.', '_stream.')

        # Load the AVI file
        video_clip = VideoFileClip(input_path)
        
        # Write the video clip to a MP4 file
        video_clip.write_videofile(output_path, codec='h264_nvenc', fps=video_clip.fps,ffmpeg_params=['-movflags', 'faststart'])
        # Close the video clip
        video_clip.close()
        
        print_and_log("Conversion successful!")
        return output_path
    except Exception as e:
        print_and_log(f"An error occurred: {e}")
        return None'''

def avi_to_mp4(input_path):
    try:
        # Determine the output path
        output_path = input_path.replace('.', '_stream.')

        # Define the ffmpeg command
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-c', 'copy',
            '-movflags', 'faststart',
            output_path
        ]

        # Execute the ffmpeg command
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the ffmpeg command was successful
        if result.returncode != 0:
            raise Exception(result.stderr)
        
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

def upload_file_to_cloud(orig_file_path, base64_sportunity_id, rec_id, server_ip, http_port):
    file_path = orig_file_path#avi_to_mp4(orig_file_path)
    print("New video path", file_path)
    '''storage_acct = "sportunitydiag304"  # Hardcoded storage account
    storage_key = "zJk05uzQ/GHKAuV2jP62BweO5FYjEFzXicuX4PiRbuBMIv9SLnYBwLIhdzmzOHjVaeMQycO4XjFhUiSelcwRaA=="  # Hardcoded storage key
    container_name = 'eventvideos'
    file_name = os.path.basename(file_path)
    print("FN", file_name)
    shared_access_policy = BlobSasPermissions(read=True)

    blob_service = BlobServiceClient(account_url=f"https://{storage_acct}.blob.core.windows.net", credential=storage_key)
    blob_client = blob_service.get_blob_client(container=container_name, blob=file_name)
    print("Opening video for blob")
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
    print_and_log(file_url)'''

    file_url = f'http://{server_ip}:{http_port}/broadcast_{rec_id}.mp4'
    print("URL", file_url)
    succ, video_id = send_video_url(file_url, base64_sportunity_id)


    '''# Delayed deletion of local file
    time.sleep(3)
    #os.remove(file_path)
    print_and_log("File deleted")
    team1_score=-1
    team2_score=-1'''


    '''video_id_str=f'Video:{video_id}'
    input_bytes = video_id_str.encode('utf-8')
    base64_bytes = base64.b64encode(input_bytes)
    base64_video_id = base64_bytes.decode('utf-8')


    cntr=0
    try:
        with open(highlight_path, 'r') as file:
            for line in file:
                # Split the line by comma to get a list of elements
                elements = line.strip().split(',')
                
                # Extract and print the first and second elements from the list
                # Assuming the format of each line is "GOAL,seconds, score"
                if len(elements) > 2:
                    print("Time (seconds):", elements[1].strip(), "Score:", elements[2].strip())
                    caption = elements[0].strip()
                    highlight_time = elements[1].strip()
                    score = elements[2].strip()
                    scores = score.split(':')
                    if len(scores) == 2:  # Make sure there are exactly two scores
                        team1_score = int(scores[0])  # Convert first part to integer
                        team2_score = int(scores[1])  # Convert second part to integer
                    add_video_highlight(base64_video_id, cntr, caption, highlight_time)
        cntr+=1
    except:
        print_and_log("Failed to open highlights file")                        

    if (team1_score!=-1):
        print_and_log("Uploading scores....")
        update_sportunity_scores(base64_sportunity_id, team1_score, team2_score)
        print_and_log("Done")
    else:
        print_and_log("There were no highlights...")'''

def send_video_url(video_url, base64_sportunity_id):
    try:
        config_backend_url = "https://backendsportunity2017.com/devices-hooks"
        username = "Sportunity"
        password = "zI4pN5wA8uK9aR1b"
        sportunity_ids = decode_and_extract_id(base64_sportunity_id)
        url = f"{config_backend_url}/store-video-url?video_url={video_url}&sportunity_ids={sportunity_ids}"

        print_and_log("Call backend URL to save the video:", url)

        response = requests.get(url, auth=(username, password))
        print_and_log("Response:", response)

        if response.status_code == 200:
            # Parse the JSON data from the response
            data = response.json()
            
            # Extract the 'success' and 'ID' fields
            success = data.get('success', False)  # Default to False if 'success' not in response
            id_value = data.get('id', 'No ID returned')  # Default to 'No ID returned' if 'ID' not in response
            
            # Use your print_and_log function to log detailed information
            print_and_log("Response success:", success)
            print_and_log("Response ID:", id_value)
            
            # Returning the extracted values if needed elsewhere
            return success, id_value    
    except:
        print("Error while calling send_video_url....")
    
    return False, 0

def run_batch_file(file_path):
    print_and_log("running ", file_path)
    try:
        process = subprocess.Popen(file_path, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE)
    except Exception as e:
        print_and_log("Error:", e)


def check_process_running(process_name, args_to_match):
    try:
        for process in psutil.process_iter():
            if process.name() == process_name:
                for arg in process.cmdline():
                    print_and_log("   ", arg,args_to_match)
                    if arg == args_to_match:
                        return True           

        print_and_log("Returning false")
    except:
        print("Error - 1")
    return False


def kill_process(process_name, args_to_match=None):
    try:
        for process in psutil.process_iter():
            if process.name() == process_name:
                if args_to_match:
                    if all(arg in process.cmdline() for arg in args_to_match):
                        process.kill()
                else:
                    process.kill()
    except:
        print("Error - 2")

class TextEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        print_and_log("Start")
        self.cache=None
        self.start_time = None   
        self.title("Sportunity VAR Scheduler")
        self.geometry("680x500")
        self.configure(bg="black")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.config = None
        self.server_config = self.load_config(SERVER_CONFIG)
        if not self.server_config:
            sys.exit(1)


        self.text_widget = tk.Text(self, wrap="word", font=("consolas", 11), bg="#000070", fg="white")
        self.text_widget.pack(expand=True, fill="both")
        self.called_ids = set()
        # Apply colors to different parts of the text
        self.text_widget.tag_configure("green", foreground="#00FF00")
        self.text_widget.tag_configure("white", foreground="#FFFF00")
        self.text_widget.tag_configure("blue", foreground="blue")
        self.text_widget.tag_configure("gray", foreground="#A0A0A0")
        self.text_widget.tag_configure("red", foreground="#FF0000")

        #self.trigger_function("U3BvcnR1bml0eTo2NjNjZTRiMzg0NGIyMjAwNDg4MmIwNTM=","24753218")
        self.refresh_text()

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()
            
    def IsRecordingOn(self, id):
        if id in self.called_ids:
            return False
        else:
            self.called_ids.add(id)
            return True        

    def refresh_text(self):
        try:
            self.text_widget.delete(1.0, "end")
            data = self.load_video_links()
            line_cntr=0
            for item in data:
                current_time = datetime.now(timezone.utc) + timedelta(hours=TIME_OFFSET)

                # Calculate remaining time in seconds
                remaining_time_seconds = (item['start_time'] - current_time).total_seconds()
                remaining_time_seconds_tmp = remaining_time_seconds

                remaining_time_to_end_seconds = (item['end_time'] - current_time).total_seconds()
                if remaining_time_to_end_seconds<0:
                    continue

                # Determine sign and absolute value of the remaining time
                if remaining_time_seconds < 0:
                    sign = "-"
                    remaining_time_seconds_tmp = abs(remaining_time_seconds_tmp)
                else:
                    sign = " "

                # Convert remaining_time_seconds to hh:mm:ss format
                hours, remainder = divmod(remaining_time_seconds_tmp, 3600)
                minutes, seconds = divmod(remainder, 60)

                # Format the result as hh:mm:ss with leading zeros if necessary
                item['remaining_time_str'] = "{}{:02}:{:02}:{:02}".format(sign, int(hours), int(minutes), int(seconds))

                rec="[  skip   ]"
                line_color="gray"
                if item['recording']:
                    if (remaining_time_seconds<=0):
                        rec="[recording]"
                        line_color="red"
                        if (self.IsRecordingOn(item['sportunityId'])):
                            self.start_recording(item['infrastructure'], item['duration'], item['sportunityId'])
                    else:
                        rec="[scheduled]"
                        line_color="green"


                self.text_widget.insert("end", "  ", "green")
                self.text_widget.insert("end", self.fix_length_string(item['start_date_str'],6), line_color)
                self.text_widget.insert("end", self.fix_length_string(item['start_time_str'],10), line_color)
                self.text_widget.insert("end", self.fix_length_string(str(item['duration'])+" min", 8), line_color)
                self.text_widget.insert("end", self.fix_length_string(str(item['remaining_time_str']), 11), line_color)


                self.text_widget.insert("end", self.fix_length_string(rec,14), line_color)
                
                self.text_widget.insert("end", self.fix_length_string(item['infrastructure'], 30)+" ", line_color)

                self.text_widget.insert("end","\n", "white")
                line_cntr+=1

            if line_cntr==0:
                self.text_widget.insert("end","No scheduled activity...\n", "white")
            height = line_cntr*20
            if height<100:
                height=100
            self.geometry(f'680x{height}')
        except:
            print("Error - 3")

        # Refresh after 1 second
        self.after(1000, self.refresh_text)


    def start_recording(self, infra, duration, su_id):
        print_and_log("Start recording...")
        self.config = self.load_config(CONFIG)
        if not self.config:
            sys.exit(1)

        field_id=2
        if infra.endswith("LS"):
            field_id=1

        self.save_changes(duration, field_id, su_id)

        kill_process(feed_process_name)
        run_batch_file(feed_batch_file)
        time.sleep(2)
        if check_process_running(feed_process_name, feed_process_name):
            print_and_log("Feed is running...")
        else:
            print_and_log("Feed is NOT running...")
            return False

        status=False
        ccnt=5
        while (ccnt>0 and status==False):
            ccnt-=1
            kill_process(ai_process_name, ai_process_args)
            run_batch_file(ai_batch_file)
            time.sleep(2)
            if check_process_running(ai_process_name, ai_process_args):
                print_and_log("AI is running...")
                status = True
                break
            else:
                print_and_log("AI is NOT running...")
                status= False
                time.sleep(5)
        
        return status

    def load_config(self, filename):
        """Load JSON configuration from a file."""
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except Exception as e:
            print_and_log(f"Failed to load configuration: {str(e)}")
            return None

    def save_config(self, filename, config):
        """Save JSON configuration to a file."""
        try:
            with open(filename, 'w') as file:
                json.dump(config, file, indent=4)
            print_and_log("Configuration saved successfully.")
        except Exception as e:
            print_and_log(f"Failed to save configuration: {str(e)}")

            self.config = self.load_config(CONFIG)
            if not self.config:
                sys.exit(1)

    def save_changes(self, duration, field_id, su_id):
        print_and_log("Duration:", duration, "FieldId:", field_id)
        self.config['source'] = "rtsp"
        self.config['mode'] = "streaming"
        self.config['field'] = field_id
        self.config['duration_min'] = duration
        self.config['sportunity'] = su_id
        self.config['starting_position_sec'] = 0
        self.config['recording_id'] = str(int(time.perf_counter() * 1000))
        self.save_config(CONFIG, self.config)
        
        threading.Timer(duration * 60+20, self.trigger_function, [su_id, self.config['recording_id'], self.server_config['server_ip'], self.server_config['http_port']]).start()

    def trigger_function(self, su_id, recording_id, server_ip, http_port):
        print_and_log(f"Triggered after timer with su_id: {su_id} and recording_id: {recording_id}")
        # Start the long operation in a new thread
        thread = threading.Thread(target=self.long_operation, args=(su_id, recording_id, server_ip, http_port))
        thread.start()

    def long_operation(self, su_id, recording_id, server_ip, http_port):
        print_and_log("Convert and upload file")
        try:
            upload_file_to_cloud(f'C:/Develop/VideoOutput/broadcast_{recording_id}.mp4', su_id, recording_id, server_ip, http_port)
        except:
            print_and_log("Error while converting video...")

    def fix_length_string(self, input_str, length):
        # Replace newline characters with " | "
        input_str = input_str.replace("\n", " | ")
        
        if len(input_str) > length:
            return input_str[:length-3] + "..."
        elif len(input_str) < length:
            return input_str + " " * (length - len(input_str))
        else:
            return input_str

    def load_video_links(self):
        try:
            if self.start_time is None or time.time() - self.start_time > 60 * 5:
                self.start_time = time.time()
            else:
                if self.cache!=None:
                    return self.cache

            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=-6)
            tomorrow = today + timedelta(days=9)

            start_time_str = today.isoformat() + "Z"
            end_time_str = tomorrow.isoformat() + "Z"

            url = GRAPHQL_URL
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            query = """
            query {
                viewer {
                    VideoLinks(VideoLinksInput:{starttime:"%s", endtime:"%s"}) {
                        id, name, start_time, end_time, urls, description, sportunityId, infrastructure, participants, organizers, teams, score
                    }
                }
            }
            """ % (start_time_str, end_time_str)

            response = requests.post(url, json={'query': query}, headers=headers)
            self.cache = []
            if response.status_code == 200:
                data = response.json()
                video_links = data['data']['viewer']['VideoLinks']
                for video_link in sorted(video_links, key=itemgetter('start_time')):
                    video_link['start_time'] = datetime.fromisoformat(video_link['start_time'].replace("Z", "+00:00")) + timedelta(hours=TIME_OFFSET)
                    video_link['end_time'] = datetime.fromisoformat(video_link['end_time'].replace("Z", "+00:00")) + timedelta(hours=TIME_OFFSET)

                    if video_link['sportunityId'] == "U3BvcnR1bml0eTo2NjNiNTY5OGUzOTFkZDAwNjc3NjUyZGQ=":
                        video_link['start_time'] = startup_time
                        video_link['end_time'] = startup_time + timedelta(seconds=60 * (STARTUP_DELAY + 2))

                    video_link['start_time'] += timedelta(minutes=STARTUP_DELAY)
                    current_time = datetime.now(timezone.utc) + timedelta(hours=TIME_OFFSET)

                    video_link['duration'] = int((video_link['end_time'] - max(video_link['start_time'],current_time)).total_seconds() / 60)
                    video_link['start_time_str'] = video_link['start_time'].strftime("%H:%M:%S")
                    video_link['start_date_str'] = video_link['start_time'].strftime("%m/%d")
                    video_link['recording'] = False
                    #print(video_link['score'])

                    if video_link['infrastructure'].strip().lower() == FIELD_NAME.strip().lower():
                        video_link['recording'] = True
                    if video_link['infrastructure'].strip().lower() in ["football 5*5 - terrain ls", "football 5*5 - terrain 2"]:
                        self.cache.append(video_link)

                self.save_video_links_to_file()

            else:
                print(f"Failed to fetch data: {response.status_code}")

            return self.cache
        except:
            print("Error - 5")

        return None


    def extract_video_url(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            video_tag = soup.find('video', id='highlightVideo')
            if video_tag is None:
                raise ValueError("No video tag with id 'highlightVideo' found in the HTML content.")
            video_source = video_tag.find('source')
            if video_source is None:
                raise ValueError("No source tag found within the video tag.")
            return video_source['src']
        except:
            return None
    def save_video_links_to_file(self):
        try:
            filename = "C:/Develop/VideoOutput/video_links.html"
            os.remove(filename)
        except:
            pass

        existing_lines = set()

        try:
            with open(filename, 'r') as file:
                for line in file:
                    existing_lines.add(line.strip())
        except FileNotFoundError:
            pass

        try:
            with open(filename, 'w') as file:
                # Write HTML header with black background and CSS for styling
                file.write("""
                <html>
                <head>
                    <title>Video Links</title>
                    <style>
                        body {
                            background-color: #202020;
                            color: white;
                            font-family: Arial, sans-serif;
                        }
                        table {
                            width: 60%;
                            border-collapse: collapse;
                        }
                        th, td {
                            border: 1px solid white;
                            padding: 10px;
                            text-align: center;
                        }
                        th {
                            background-color: #333;
                        }
                        .upcoming {
                            color: yellow;
                        }
                        .available {
                            color: lightgreen;
                        }
                        .not-available {
                            color: red;
                        }
                        .field-ls {
                            color: lightgreen;
                        }
                        .field-2 {
                            color: lightblue;
                        }
                        tr:nth-child(even) {
                            background-color: #283838;
                        }
                    </style>
                </head>
                <body>
                <table>
                <tr>
                    <th>Field</th>
                    <th>Start Time</th>
                    <th>Duration</th>
                    <th>Score</th>
                    <th>Online Link</th>
                    <th>mp4 Link</th>
                </tr>
                """)

                for video_link in self.cache:
                    if video_link['infrastructure'].strip().lower() == "football 5*5 - terrain ls":
                        port = 8090
                        field_class = "field-ls"
                    elif video_link['infrastructure'].strip().lower() == "football 5*5 - terrain 2":
                        port = 8092
                        field_class = "field-2"
                    else:
                        continue

                    start_time_str = video_link['start_time'].strftime("%Y-%m-%d %H:%M:%S")
                    duration = video_link['duration']
                    score = video_link.get('score', 'N/A')
                    sportunityid = video_link['sportunityId']
                    url = f"https://lahalle.sportunity.com:{port}/{sportunityid}"

                    video_start_time = video_link['start_time']-timedelta(hours=TIME_OFFSET)
                    current_time = datetime.now(video_start_time.tzinfo)  # Make current time timezone-aware
                    video_end_time = video_start_time + timedelta(minutes=duration)

                    file_exists=False
                    file_path=None
                    if video_start_time < current_time:
                        file_path = f"C:\\Develop\\VideoOutput\\{sportunityid}.html"
                        file_exists = os.path.isfile(file_path)
                        if not file_exists:
                            file_path = f"Z:\\{sportunityid}.html"
                            file_exists = os.path.isfile(file_path)
                        
                        if file_exists:
                            online_link = f'<a href="{url}" target="_blank" class="available">Watch</a>'
                            link_mp4 = self.extract_video_url(file_path)
                            if link_mp4 is None:
                                link_mp4 = "n/a"
                            else:
                                link_mp4 = f'<a href="{link_mp4}" target="_blank" class="available">Open mp4</a>'
                        else:
                            online_link = '<span class="not-available">Not available</span>'
                            link_mp4 = '<span class="not-available">Not available</span>'
                    else:
                        time_to_start = (video_start_time - current_time).total_seconds()
                        online_link = f'<span class="upcoming">Starting in {int(time_to_start // 3600)}h {int((time_to_start % 3600) // 60)}m {int(time_to_start % 60)}s</span>'
                        link_mp4 = '<span class="upcoming">Upcoming</span>'

                    if current_time >= video_start_time and file_exists:
                        online_link = f'<a href="{url}" target="_blank" class="available">Watch</a>'
                        if current_time >= video_end_time:
                            link_mp4 = self.extract_video_url(file_path)
                            if link_mp4 is None:
                                link_mp4 = "n/a"
                            else:
                                link_mp4 = f'<a href="{link_mp4}" target="_blank" class="available">Open mp4</a>'

                    line = f"<tr><td class='{field_class}'>{video_link['infrastructure']}</td><td>{start_time_str}</td><td>{duration} mins</td><td>{score}</td><td>{online_link}</td><td>{link_mp4}</td></tr>"

                    if line not in existing_lines:
                        file.write(line + '\n')

                # Write HTML footer
                file.write("""
                </table>
                </body>
                </html>
                """)
        except:
            print("error - 8")
            
import socket

if __name__ == "__main__":

    port=8293
    # Create monitoring socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', port))
    s.listen(5)
    s.setblocking(False)  # Set the socket to non-blocking mode

    print("Listening on port port...")    
    #add_video_highlight("VmlkZW86NjY0MTRlZTVhNmU2NDgwMDY3YTY1MThh",4,"GOAL",122)
    #obj_id = ObjectId(id_string)
    #id="66414ee5a6e6480067a6518a"
    #video_id_str=f'Video:{id}'
    #input_bytes = video_id_str.encode('utf-8')
    #base64_bytes = base64.b64encode(input_bytes)
    #base64_string = base64_bytes.decode('utf-8')
    #print(base64_string)
    #update_sportunity_scores("U3BvcnR1bml0eTo2NjNiNTY5OGUzOTFkZDAwNjc3NjUyZGQ=",3,2)
    if len(sys.argv) > 1:
        FIELD_NAME = sys.argv[1]
        app = TextEditor()
        app.mainloop()
    else:
        print("No parameters provided.")    
