import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys

class TeamIdentifier:
    """
    A class to identify team membership based on uniform colors
    """
    def __init__(self, num_teams=2):
        self.num_teams = num_teams
        self.team_colors = None  # Will store representative colors for each team
        self.team_assignments = {}  # player_id -> team_id
        self.team_display_colors = None  # Colors to use for displaying teams
        self.initialized = False
    
    def extract_dominant_colors(self, image, k=3):
        """Extract dominant colors from a player image"""
        # Reshape image for color analysis
        pixels = image.reshape(-1, 3)
        
        # Filter out black/near-black pixels (often background or shadows)
        non_black_mask = np.sum(pixels, axis=1) > 30
        filtered_pixels = pixels[non_black_mask]
        
        # If we have too few pixels after filtering, return None
        if len(filtered_pixels) < 100:
            return None
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get the colors and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        
        # Sort colors by frequency
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = colors[sorted_indices]
        
        return sorted_colors
    
    def extract_shirt_color(self, player_img, keypoints=None):
        """
        Extract the primary shirt color from player image
        If keypoints are provided, focus on torso area
        """
        # Default: use the upper third of the image for shirt
        h, w = player_img.shape[:2]
        shirt_region = player_img[0:h//3, :]
        
        # If we have pose keypoints, use them to get a better shirt region
        if keypoints and len(keypoints) > 10:  # Make sure we have enough keypoints
            # Typical indices for shoulders and hips in COCO format
            left_shoulder = keypoints[5] if len(keypoints) > 5 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 else None
            left_hip = keypoints[11] if len(keypoints) > 11 else None
            right_hip = keypoints[12] if len(keypoints) > 12 else None
            
            # If we have valid shoulder and hip points
            if (left_shoulder and right_shoulder and left_hip and right_hip and
                all(point is not None and len(point) >= 2 for point in [left_shoulder, right_shoulder, left_hip, right_hip])):
                
                # Calculate torso region
                top = min(left_shoulder[1], right_shoulder[1])
                bottom = max(left_hip[1], right_hip[1])
                left = min(left_shoulder[0], left_hip[0])
                right = max(right_shoulder[0], right_hip[0])
                
                # Ensure values are within image bounds
                top = max(0, top)
                bottom = min(h, bottom)
                left = max(0, left)
                right = min(w, right)
                
                # Extract torso region
                if bottom > top and right > left:
                    shirt_region = player_img[top:bottom, left:right]
        
        # Get dominant colors
        colors = self.extract_dominant_colors(shirt_region, k=2)
        return colors[0] if colors is not None and len(colors) > 0 else None
    
    def initialize_teams(self, player_images):
        """
        Analyze player images to determine team colors
        
        Args:
            player_images: Dict of player_id -> (image, keypoints)
        """
        print("Initializing team identification...")
        
        # Extract shirt colors for each player
        player_colors = {}
        for player_id, (image, keypoints) in player_images.items():
            shirt_color = self.extract_shirt_color(image, keypoints)
            if shirt_color is not None:
                player_colors[player_id] = shirt_color
        
        if len(player_colors) < self.num_teams:
            print(f"Warning: Not enough players with valid colors ({len(player_colors)}) to identify {self.num_teams} teams")
            return False
        
        # Use K-means to cluster players into teams based on shirt colors
        colors_array = np.array(list(player_colors.values()))
        kmeans = KMeans(n_clusters=self.num_teams, n_init=10)
        team_labels = kmeans.fit_predict(colors_array)
        
        # Store team representative colors (cluster centers)
        self.team_colors = kmeans.cluster_centers_.astype(int)
        
        # Assign players to teams
        for i, player_id in enumerate(player_colors.keys()):
            self.team_assignments[player_id] = int(team_labels[i])
        
        # Set display colors for teams (distinct colors for visualization)
        self.team_display_colors = [
            (0, 0, 255),    # Red team (BGR format)
            (0, 255, 0),    # Green team
            (255, 0, 0),    # Blue team
            (0, 255, 255),  # Yellow team
            (255, 0, 255),  # Magenta team
        ]
        
        # Log team colors
        print("Team identification complete!")
        for i, color in enumerate(self.team_colors):
            print(f"Team {i+1} color: BGR{tuple(color)} - Display color: BGR{self.team_display_colors[i]}")
        
        self.initialized = True
        return True
    
    def get_team_for_player(self, player_id, player_image=None):
        """
        Get the team assignment for a player
        
        If player not previously assigned and image provided, analyze and assign
        """
        # If player already assigned, return team
        if player_id in self.team_assignments:
            return self.team_assignments[player_id]
        
        # If no teams initialized or no image provided, return -1 (unknown)
        if not self.initialized or player_image is None:
            return -1
        
        # Extract shirt color and find nearest team color
        shirt_color = self.extract_shirt_color(player_image)
        if shirt_color is None:
            return -1
        
        # Find closest team color
        distances = [np.linalg.norm(shirt_color - team_color) for team_color in self.team_colors]
        closest_team = np.argmin(distances)
        
        # Store assignment for future use
        self.team_assignments[player_id] = closest_team
        return closest_team
    
    def get_display_color_for_team(self, team_id):
        """Get the display color for a team"""
        if not self.initialized or team_id < 0 or team_id >= self.num_teams:
            return (180, 180, 180)  # Gray for unknown team
        
        return self.team_display_colors[team_id]
    
    def get_display_color_for_player(self, player_id, player_image=None):
        """Get the display color for a player based on team"""
        team_id = self.get_team_for_player(player_id, player_image)
        return self.get_display_color_for_team(team_id)