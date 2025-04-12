import pychromecast
import time

def list_available_chromecasts():
    # Discover all Chromecast devices
    chromecasts, browser = pychromecast.get_chromecasts()
    if not chromecasts:
        print("No Chromecast devices found.")
        return

    print("Available Chromecast devices:")
    for cast in chromecasts:
        print(f" - {cast.name}")

def cast_video_to_chromecast(chromecast_name, video_file):
    # Discover Chromecast devices with the specified name
    chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=[chromecast_name])
    if not chromecasts:
        print(f"No Chromecast with name '{chromecast_name}' found.")
        return

    # Connect to the Chromecast
    cast = chromecasts[0]
    cast.wait()

    # Get the media controller
    mc = cast.media_controller

    # Load and play the video file
    mc.play_media(video_file, 'video/mp4')
    mc.block_until_active()

    print(f"Playing '{video_file}' on '{chromecast_name}'")

    # Keep the script running while the video is playing
    while mc.status.player_state != 'IDLE':
        time.sleep(1)

    print("Playback finished")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cast_video.py <command> [<chromecast_name> <video_file>]")
        print("Commands:")
        print("  list             List available Chromecast devices")
        print("  cast <name> <url>  Cast the specified video URL to the specified Chromecast")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        list_available_chromecasts()
    elif command == "cast" and len(sys.argv) == 4:
        chromecast_name = sys.argv[2]
        video_file = sys.argv[3]
        cast_video_to_chromecast(chromecast_name, video_file)
    else:
        print("Invalid command or missing arguments.")
        print("Usage: python cast_video.py <command> [<chromecast_name> <video_file>]")
        print("Commands:")
        print("  list             List available Chromecast devices")
        print("  cast <name> <url>  Cast the specified video URL to the specified Chromecast")
        sys.exit(1)
