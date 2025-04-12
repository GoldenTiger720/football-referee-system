import subprocess
import json
import time
import psutil
import pytz
import datetime

def run_batch_file(file_path):
    print("running ", file_path)
    subprocess.Popen(file_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def check_process_running(process_name, args_to_match):
    for process in psutil.process_iter():
        if process.name() == process_name:
            for arg in process.cmdline():
                print("   ", arg)
            if args_to_match:
                if all(arg in process.cmdline() for arg in args_to_match): 
                    print("returning true")           
                    return True
    print("Returning false")
    return False

def kill_process(process_name, args_to_match=None):
    for process in psutil.process_iter():
        if process.name() == process_name:
            if args_to_match:
                if all(arg in process.cmdline() for arg in args_to_match):
                    process.kill()
            else:
                process.kill()

def main():
    scheduler_file = "scheduler.json"
    feed_batch_file = r"C:\Develop\Configuration\start_feed.bat"
    ai_batch_file = r"C:\Develop\Configuration\start_ai.bat"
    feed_process_name = "CameraFeeds.exe"
    ai_process_name = "python.exe"
    ai_process_args = ["main3.py"]

    while True:
        with open(scheduler_file, 'r') as f:
            data = json.load(f)

        start_time_str = data[0]["start_time"]
        start_time = datetime.datetime.strptime(start_time_str, "%Y.%m.%d %H:%M")
        current_time = datetime.datetime.now()
        delta = start_time - current_time
        delta_seconds = delta.total_seconds()

        print(delta_seconds)

        if not check_process_running(feed_process_name, ""):
            print("Feed not running. Starting...")
            #kill_process(feed_process_name)
            run_batch_file(feed_batch_file)
        #else:
        #    kill_process(feed_process_name)

        #if not check_process_running(ai_process_name, ["scheduler.py"]):
        #    print("AI not running")
        #    run_batch_file(ai_batch_file)
        #else:
        #    kill_process(ai_process_name, ai_process_args)

      
        time.sleep(2)  # Check every minute

if __name__ == "__main__":
    main()
