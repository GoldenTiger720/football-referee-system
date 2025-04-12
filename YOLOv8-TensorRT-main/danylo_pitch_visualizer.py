import tkinter as tk
import math
import time
from collections import deque

class PitchVisualizer:
    """
    Visualizes the soccer pitch with player and ball positions,
    as well as events and statistics.
    """
    
    def __init__(self, canvas):
        """
        Initialize the pitch visualizer with a canvas
        
        Args:
            canvas: Tkinter canvas to draw on
        """
        self.canvas = canvas
        
        # Colors
        self.ball_color = "red"
        self.ball_trail_color = "orange"
        self.player_color = "blue"
        self.player_highlight_color = "green"
        self.text_color = "white"
        
        # Visualization parameters
        self.ball_radius = 5
        self.player_radius = 8
        self.show_labels = True
        self.show_ball_trail = True
        self.show_path_prediction = True
        
        # Ball trail
        self.ball_trail = deque(maxlen=20)
        
        # Objects on canvas
        self.canvas_objects = {
            'ball': None,
            'ball_trail': [],
            'players': {},
            'labels': {},
            'events': [],
            'stats': {},
            'paths': []
        }
        
        # Previous dimensions for resize handling
        self.prev_width = self.canvas.winfo_width()
        self.prev_height = self.canvas.winfo_height()
        
        # Force initial resize
        self.canvas.after(100, self.check_resize)
    
    def check_resize(self):
        """Check if canvas has been resized and update accordingly"""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width != self.prev_width or height != self.prev_height:
            self.prev_width = width
            self.prev_height = height
            self.clear_all()
            
        # Schedule next check
        self.canvas.after(100, self.check_resize)
    
    def clear_all(self):
        """Clear all objects from the canvas"""
        self.canvas.delete("all")
        
        # Reset object tracking
        for key in self.canvas_objects:
            if isinstance(self.canvas_objects[key], dict):
                self.canvas_objects[key] = {}
            elif isinstance(self.canvas_objects[key], list):
                self.canvas_objects[key] = []
            else:
                self.canvas_objects[key] = None
    
    def update(self, analysis_results):
        """
        Update the pitch visualization with new analysis results
        
        Args:
            analysis_results: Results from MatchAnalyzer
        """
        if not analysis_results:
            return
            
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            # Canvas not yet properly sized
            return
        
        # Clear previous visualizations
        self.clear_all()
        
        # Draw ball
        if 'ball_tracking' in analysis_results and analysis_results['ball_tracking']:
            self.draw_ball(analysis_results['ball_tracking'], width, height)
        
        # Draw players
        if 'player_tracking' in analysis_results:
            for player_data in analysis_results['player_tracking']:
                self.draw_player(player_data, width, height)
        
        # Draw events
        if 'events' in analysis_results:
            self.draw_events(analysis_results['events'], width, height)
        
        # Draw stats
        if 'score' in analysis_results and 'match_time' in analysis_results:
            self.draw_stats(analysis_results, width, height)
    
    def convert_coords(self, x_2d, y_2d, width, height):
        """
        Convert pitch coordinates (0-500 x, 0-250 y) to canvas coordinates
        
        Args:
            x_2d, y_2d: Pitch coordinates
            width, height: Canvas dimensions
            
        Returns:
            Canvas coordinates (x, y)
        """
        # Apply margins (10% on each side)
        margin_x = width * 0.1
        margin_y = height * 0.1
        
        # Scale coordinates
        pitch_width = 500  # Default pitch width in units
        pitch_height = 250  # Default pitch height in units
        
        # Calculate drawable area
        drawable_width = width - 2 * margin_x
        drawable_height = height - 2 * margin_y
        
        # Scale while maintaining aspect ratio
        aspect_ratio = pitch_width / pitch_height
        canvas_aspect_ratio = drawable_width / drawable_height
        
        if canvas_aspect_ratio > aspect_ratio:
            # Canvas is wider than needed
            scale_factor = drawable_height / pitch_height
            used_width = pitch_width * scale_factor
            x_offset = margin_x + (drawable_width - used_width) / 2
            y_offset = margin_y
        else:
            # Canvas is taller than needed
            scale_factor = drawable_width / pitch_width
            used_height = pitch_height * scale_factor
            x_offset = margin_x
            y_offset = margin_y + (drawable_height - used_height) / 2
        
        # Convert coordinates
        canvas_x = x_offset + x_2d * scale_factor
        canvas_y = y_offset + y_2d * scale_factor
        
        return canvas_x, canvas_y
    
    def draw_ball(self, ball_data, width, height):
        """Draw the ball and its trail on the canvas"""
        x_2d = ball_data['x_2d']
        y_2d = ball_data['y_2d']
        
        # Convert to canvas coordinates
        canvas_x, canvas_y = self.convert_coords(x_2d, y_2d, width, height)
        
        # Draw ball
        self.canvas_objects['ball'] = self.canvas.create_oval(
            canvas_x - self.ball_radius, 
            canvas_y - self.ball_radius,
            canvas_x + self.ball_radius, 
            canvas_y + self.ball_radius,
            fill=self.ball_color,
            outline="white",
            width=1,
            tags=("ball",)
        )
        
        # Add to trail
        self.ball_trail.append((canvas_x, canvas_y))
        
        # Draw trail
        if self.show_ball_trail:
            # Create gradient of colors for trail
            trail_colors = []
            for i in range(len(self.ball_trail)):
                # Fade from ball color to transparent
                alpha = 1.0 - (i / len(self.ball_trail))
                r = int(255 * alpha)
                g = int(165 * alpha)
                b = int(0)
                trail_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            
            # Draw trail segments
            for i in range(len(self.ball_trail) - 1):
                if i < len(trail_colors):
                    x1, y1 = self.ball_trail[i]
                    x2, y2 = self.ball_trail[i+1]
                    
                    # Calculate segment radius based on position
                    radius = self.ball_radius * (1.0 - (i / len(self.ball_trail)))
                    
                    # Draw trail segment as a line
                    line_id = self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill=trail_colors[i],
                        width=radius * 2,
                        capstyle=tk.ROUND,
                        joinstyle=tk.ROUND,
                        tags=("ball_trail",)
                    )
                    
                    self.canvas_objects['ball_trail'].append(line_id)
        
        # Draw speed and direction indicators if available
        if 'speed_kmh' in ball_data and ball_data['speed_kmh'] > 0:
            # Add speed text
            speed_text = f"{ball_data['speed_kmh']:.1f} km/h"
            speed_text_id = self.canvas.create_text(
                canvas_x, canvas_y - self.ball_radius - 10,
                text=speed_text,
                fill=self.text_color,
                font=("Arial", 8),
                tags=("ball_speed",)
            )
            self.canvas_objects['labels']['ball_speed'] = speed_text_id
            
            # Draw direction arrow if direction is available
            if 'direction' in ball_data and ball_data['direction'] >= 0:
                direction_rad = math.radians(ball_data['direction'])
                arrow_length = min(20, ball_data['speed_kmh'] * 0.5)  # Scale arrow by speed
                
                # Calculate end point
                arrow_x = canvas_x + arrow_length * math.cos(direction_rad)
                arrow_y = canvas_y + arrow_length * math.sin(direction_rad)
                
                # Draw arrow
                arrow_id = self.canvas.create_line(
                    canvas_x, canvas_y,
                    arrow_x, arrow_y,
                    fill=self.ball_color,
                    width=2,
                    arrow=tk.LAST,
                    tags=("ball_direction",)
                )
                self.canvas_objects['labels']['ball_direction'] = arrow_id
    
    def draw_player(self, player_data, width, height):
        """Draw a player on the canvas"""
        x_2d = player_data['x_2d']
        y_2d = player_data['y_2d']
        player_id = player_data['player_id']
        
        # Convert to canvas coordinates
        canvas_x, canvas_y = self.convert_coords(x_2d, y_2d, width, height)
        
        # Determine player color
        color = self.player_highlight_color if player_data.get('closest_to_ball', False) else self.player_color
        
        # Draw player
        player_id_str = str(player_id)
        if player_id_str not in self.canvas_objects['players']:
            self.canvas_objects['players'][player_id_str] = {}
        
        # Draw player circle
        self.canvas_objects['players'][player_id_str]['circle'] = self.canvas.create_oval(
            canvas_x - self.player_radius, 
            canvas_y - self.player_radius,
            canvas_x + self.player_radius, 
            canvas_y + self.player_radius,
            fill=color,
            outline="white",
            width=1,
            tags=("player", f"player_{player_id}")
        )
        
        # Draw player label if enabled
        if self.show_labels:
            self.canvas_objects['players'][player_id_str]['label'] = self.canvas.create_text(
                canvas_x, canvas_y,
                text=str(player_id),
                fill="white",
                font=("Arial", 8, "bold"),
                tags=("player_label", f"player_label_{player_id}")
            )
    
    def draw_events(self, events, width, height):
        """Draw event indicators on the canvas"""
        for i, event in enumerate(events):
            event_type = event.get('type', '')
            
            if event_type in ['kick', 'player_kick'] and 'position' in event:
                # Draw kick indicator
                x_2d, y_2d = event['position']
                canvas_x, canvas_y = self.convert_coords(x_2d, y_2d, width, height)
                
                # Draw burst effect
                burst_radius = 15
                event_id = self.canvas.create_oval(
                    canvas_x - burst_radius,
                    canvas_y - burst_radius,
                    canvas_x + burst_radius,
                    canvas_y + burst_radius,
                    outline="yellow",
                    width=2,
                    tags=("event", "kick")
                )
                self.canvas_objects['events'].append(event_id)
                
                # Animate burst (pulsing effect)
                self.animate_burst(event_id, 0)
                
            elif event_type in ['goal_left', 'goal_right']:
                # Draw goal indicator
                self.draw_goal_indicator(event, width, height)
    
    def animate_burst(self, item_id, step):
        """Animate a burst effect by scaling it over time"""
        # Check if object still exists
        if step < 10 and self.canvas.winfo_exists():
            try:
                # Scale effect
                scale_factor = 1.0 + (step * 0.1)
                self.canvas.scale(item_id, 0, 0, scale_factor, scale_factor)
                
                # Reduce opacity
                opacity = int(255 * (1.0 - (step / 10)))
                self.canvas.itemconfig(item_id, outline=f"#ffff{opacity:02x}")
                
                # Schedule next animation step
                self.canvas.after(50, lambda: self.animate_burst(item_id, step + 1))
            except:
                # Object may have been deleted
                pass
    
    def draw_goal_indicator(self, event, width, height):
        """Draw a goal indicator"""
        # Determine which side the goal was scored
        if event['type'] == 'goal_left':
            x_2d = 25  # Left goal area
            text_x = 100
        else:
            x_2d = 475  # Right goal area
            text_x = 400
            
        y_2d = 125  # Center of goal height
        
        # Convert to canvas coordinates
        canvas_x, canvas_y = self.convert_coords(x_2d, y_2d, width, height)
        text_canvas_x, text_canvas_y = self.convert_coords(text_x, y_2d, width, height)
        
        # Draw goal text
        goal_text = f"GOAL! Score: {event['score_left']} - {event['score_right']}"
        goal_text_id = self.canvas.create_text(
            text_canvas_x, text_canvas_y,
            text=goal_text,
            fill="yellow",
            font=("Arial", 14, "bold"),
            tags=("event", "goal_text")
        )
        self.canvas_objects['events'].append(goal_text_id)
        
        # Draw goal flash
        flash_size = 30
        goal_flash_id = self.canvas.create_rectangle(
            canvas_x - flash_size,
            canvas_y - flash_size,
            canvas_x + flash_size,
            canvas_y + flash_size,
            fill="yellow",
            outline="",
            tags=("event", "goal_flash")
        )
        self.canvas_objects['events'].append(goal_flash_id)
        
        # Animate goal flash
        self.animate_goal_flash(goal_flash_id, 0)
    
    def animate_goal_flash(self, item_id, step):
        """Animate a goal flash effect"""
        # Check if object still exists
        if step < 10 and self.canvas.winfo_exists():
            try:
                # Fade flash
                opacity = int(255 * (1.0 - (step / 10)))
                self.canvas.itemconfig(item_id, fill=f"#ffff00{opacity:02x}")
                
                # Schedule next animation step
                self.canvas.after(100, lambda: self.animate_goal_flash(item_id, step + 1))
            except:
                # Object may have been deleted
                pass
    
    def draw_stats(self, results, width, height):
        """Draw match statistics on the canvas"""
        # Score display
        score_left = results['score']['left']
        score_right = results['score']['right']
        score_text = f"{score_left} - {score_right}"
        
        # Match time
        match_time = results['match_time']
        
        # Possession stats if available
        possession_text = ""
        if 'possession' in results:
            left_percent = results['possession'].get('left', 50)
            right_percent = results['possession'].get('right', 50)
            possession_text = f"Possession: {left_percent}% - {right_percent}%"
        
        # Draw stats at top of canvas
        stats_y = height * 0.05  # 5% from top
        
        # Draw match time
        time_id = self.canvas.create_text(
            width / 2, stats_y,
            text=match_time,
            fill=self.text_color,
            font=("Arial", 14, "bold"),
            tags=("stats", "match_time")
        )
        self.canvas_objects['stats']['match_time'] = time_id
        
        # Draw score
        score_id = self.canvas.create_text(
            width / 2, stats_y + 25,
            text=score_text,
            fill=self.text_color,
            font=("Arial", 18, "bold"),
            tags=("stats", "score")
        )
        self.canvas_objects['stats']['score'] = score_id
        
        # Draw possession if available
        if possession_text:
            possession_id = self.canvas.create_text(
                width / 2, stats_y + 50,
                text=possession_text,
                fill=self.text_color,
                font=("Arial", 12),
                tags=("stats", "possession")
            )
            self.canvas_objects['stats']['possession'] = possession_id