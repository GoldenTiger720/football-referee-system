import csv
import os
from datetime import datetime

class PlayerPositionLogger:
    """
    A class to log player positions to a CSV file during football video analysis.
    """
    def __init__(self, output_dir="player_logs"):
        """
        Initialize the logger with an output directory.
        
        Args:
            output_dir: Directory where CSV files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(output_dir, f"player_positions_{timestamp}.csv")
        
        # Initialize the CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'cycle', 'position', 'player_id', 'x1', 'y1', 'x2', 'y2', 
                'x_2d', 'y_2d', 'confidence', 'cam_id', 
                'ball_distance', 'ball_x', 'ball_y', 'closest', 'timestamp'
            ])
        
        self.position_count = 0
        print(f"Player position logger initialized. Output file: {self.csv_filename}")
    
    def log_positions(self, cycle, position, players, timestamp=None):
        """
        Log player positions to the CSV file.
        
        Args:
            cycle: The detection cycle
            position: The position within the cycle
            players: List of player objects with position data
            timestamp: Optional timestamp (will use current time if None)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for player in players:
                # Extract player data, handling potential missing attributes
                player_id = getattr(player, 'player_id', -1)
                x1 = getattr(player, 'x1', -1)
                y1 = getattr(player, 'y1', -1)
                x2 = getattr(player, 'x2', -1)
                y2 = getattr(player, 'y2', -1)
                x_2d = getattr(player, 'x_2d', -1)
                y_2d = getattr(player, 'y_2d', -1)
                confidence = getattr(player, 'confidence', -1)
                cam_id = getattr(player, 'cam_id', -1)
                ball_distance = getattr(player, 'ball_distance', -1)
                ball_x = getattr(player, 'ball_x', -1)
                ball_y = getattr(player, 'ball_y', -1)
                closest = getattr(player, 'closest', False)
                
                writer.writerow([
                    cycle, position, player_id, x1, y1, x2, y2, 
                    x_2d, y_2d, confidence, cam_id, 
                    ball_distance, ball_x, ball_y, closest, timestamp
                ])
                
                self.position_count += 1
        
        return len(players)
    
    def get_stats(self):
        """
        Return statistics about the logged positions.
        
        Returns:
            Dict with stats about the logging
        """
        return {
            'csv_file': self.csv_filename,
            'position_count': self.position_count
        }