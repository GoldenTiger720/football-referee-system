import csv
import os
import time
from datetime import datetime

def collect_player_positions(output_file='player_positions.csv'):
    """
    A standalone script to collect player positions from the existing code.
    This function should be called from the process_detection_results function.
    
    Args:
        output_file: Path to the output CSV file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'timestamp', 'cycle', 'position', 'player_id', 'cam_id',
                'x1', 'y1', 'x2', 'y2', 'x_2d', 'y_2d', 
                'confidence', 'ball_distance', 'ball_x', 'ball_y', 'closest'
            ])
        
        # Current timestamp for all records in this batch
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Return the writer object so it can be used in the main code
        return writer, timestamp

def save_player_position(writer, timestamp, cycle, position, player):
    """
    Save a single player's position data.
    
    Args:
        writer: CSV writer object
        timestamp: Current timestamp
        cycle: Detection cycle
        position: Position in cycle
        player: Player object with position data
    """
    # Extract player data with defaults for missing attributes
    player_id = getattr(player, 'player_id', -1)
    cam_id = getattr(player, 'cam_id', -1)
    x1 = getattr(player, 'x1', -1)
    y1 = getattr(player, 'y1', -1)
    x2 = getattr(player, 'x2', -1)
    y2 = getattr(player, 'y2', -1)
    x_2d = getattr(player, 'x_2d', -1)
    y_2d = getattr(player, 'y_2d', -1)
    confidence = getattr(player, 'confidence', -1)
    ball_distance = getattr(player, 'ball_distance', -1)
    ball_x = getattr(player, 'ball_x', -1)
    ball_y = getattr(player, 'ball_y', -1)
    closest = getattr(player, 'closest', False)
    
    writer.writerow([
        timestamp, cycle, position, player_id, cam_id,
        x1, y1, x2, y2, x_2d, y_2d, 
        confidence, ball_distance, ball_x, ball_y, closest
    ])

# Example of how to integrate this into the process_detection_results function:
"""
# Inside process_detection_results:

# Initialize CSV writer for player positions
csv_writer, timestamp = collect_player_positions(
    os.path.join(ai_settings.video_out_folder, f'player_positions_{ai_settings.recording_id}.csv')
)

# When processing players:
for player in players_collection:
    if player.x1 != -1:
        # Process player as normal...
        
        # Save player position
        save_player_position(csv_writer, timestamp, cycle, cam.camera_id, player)
"""