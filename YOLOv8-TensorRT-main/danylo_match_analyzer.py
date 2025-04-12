import numpy as np
import math
import time
from collections import deque, defaultdict

class MatchAnalyzer:
    """
    Analyzes soccer match events based on detection results.
    Processes player and ball tracking, detects events like kicks and goals.
    """
    def __init__(self):
        """Initialize match analyzer with tracking data structures"""
        # Match state
        self.score_left = 0
        self.score_right = 0
        self.match_time = 0  # in seconds
        self.match_time_str = "00:00"
        
        # Ball tracking
        self.ball_history = deque(maxlen=60)  # Last 60 frames of ball positions
        self.ball_speed_queue = deque(maxlen=6)  # For moving average
        self.ball_accel_queue = deque(maxlen=6)  # For moving average
        self.ball_possession = {
            'left': 0,
            'right': 0
        }
        self.previous_ball_speed = -1
        self.last_ball_direction = 0
        self.last_valid_ball_x = 0
        self.last_valid_ball_y = 0
        
        # Event detection
        self.kick_detected = False
        self.kick_ttl = 0
        self.goal_hold_back_frames = 30
        self.goal_hold_back_left = -1
        self.goal_hold_back_right = -1
        
        # Player tracking
        self.player_tracking = {}  # Player ID -> tracking data
        self.next_player_id = 1
        self.player_appearance_counts = defaultdict(int)
        
        # Current action
        self.current_action = None
        self.action_history = []
        self.max_actions = 10
        
        # Goal detection parameters
        self.goal_line_left_x = 50  # X coordinate of left goal line (pitch coordinates)
        self.goal_line_right_x = 450  # X coordinate of right goal line (pitch coordinates)
        self.goal_top_y = 125  # Y coordinate of goal top (pitch coordinates)
        self.goal_bottom_y = 175  # Y coordinate of goal bottom (pitch coordinates)
        
        # Stats
        self.frames_processed = 0
        self.ball_detected_frames = 0
        self.player_detected_frames = 0

    def analyze_frame(self, frame_idx, detection_results):
        """
        Analyze a frame with detection results to identify events
        
        Args:
            frame_idx: Frame index (used for timing)
            detection_results: List of detection results from DetectionEngine
            
        Returns:
            Analysis results including tracking data and events
        """
        # Update match time (assuming 25 FPS)
        self.match_time = frame_idx / 25.0
        minutes = int(self.match_time / 60)
        seconds = int(self.match_time % 60)
        self.match_time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Initialize frame results
        result = {
            'frame_idx': frame_idx,
            'match_time': self.match_time_str,
            'ball_tracking': None,
            'player_tracking': [],
            'events': [],
            'score': {
                'left': self.score_left,
                'right': self.score_right
            },
            'possession': {
                'left': self.ball_possession['left'],
                'right': self.ball_possession['right']
            }
        }
        
        # Increment frame counter
        self.frames_processed += 1
        
        # Process ball detections
        ball_detected = False
        for detection in detection_results:
            if detection['class'] in ['ball', 'player']:
                self.process_ball_detection(detection, result)
                ball_detected = True
        
        if ball_detected:
            self.ball_detected_frames += 1
            
            # Check for goals
            self.check_for_goals(result)
            
            # Update possession stats
            self.update_possession_stats()
        
        # Process player detections
        player_detected = False
        for detection in detection_results:
            if detection['class'] == 'player':
                self.process_player_detection(detection, result)
                player_detected = True
        
        if player_detected:
            self.player_detected_frames += 1
        
        # Find closest player to ball
        if ball_detected and player_detected:
            self.find_closest_player_to_ball(result)
        
        return result
    
    def process_ball_detection(self, ball_detection, result):
        """Process a ball detection, tracking position, speed, and direction"""
        # Extract ball data
        center_x = ball_detection['center_x']
        center_y = ball_detection['center_y']
        width = ball_detection['width']
        height = ball_detection['height']
        
        # Calculate 2D pitch coordinates (0-500 x, 0-250 y)
        # Assuming the video frame is scaled to match the detection engine's input
        # and the pitch is represented in a 500x250 coordinate system
        x_2d = int(center_x / 640 * 500)  # Scale to pitch coords
        y_2d = int(center_y / 384 * 250)  # Scale to pitch coords
        
        # Calculate ball radius
        radius = min(width, height) / 2
        
        # Calculate ball speed using distance from last position
        ball_speed_kmh = -1
        ball_accel = -1
        ball_direction = -1
        
        # If we have previous ball positions
        if self.ball_history:
            last_ball_data = self.ball_history[-1]
            last_x_2d = last_ball_data['x_2d']
            last_y_2d = last_ball_data['y_2d']
            
            # Calculate distance in pitch coordinates
            distance_px = math.sqrt((x_2d - last_x_2d) ** 2 + (y_2d - last_y_2d) ** 2)
            
            # Convert to real-world distance (assuming 500px = 25m width)
            distance_m = (distance_px / 500) * 25
            
            # Calculate time between frames (assuming 25 FPS)
            time_s = 1 / 25
            
            # Calculate speed in m/s and km/h
            speed_ms = distance_m / time_s
            speed_kmh = speed_ms * 3.6
            
            # Apply moving average
            ball_speed_kmh = self.calculate_moving_average(self.ball_speed_queue, speed_kmh)
            
            # Calculate acceleration
            if self.previous_ball_speed > 0:
                v1_ms = self.previous_ball_speed * (1000 / 3600)
                v2_ms = ball_speed_kmh * (1000 / 3600)
                delta_v_ms = v2_ms - v1_ms
                delta_t_s = 0.04  # 40 milliseconds (1/25 sec)
                
                accel = delta_v_ms / delta_t_s
                ball_accel = self.calculate_moving_average(self.ball_accel_queue, accel)
            
            # Calculate ball direction
            dy = center_y - self.last_valid_ball_y
            dx = center_x - self.last_valid_ball_x
            angle_radians = math.atan2(dy, dx)
            angle_degrees = math.degrees(angle_radians)
            ball_direction = (angle_degrees + 360) % 360
            
            # Check for direction change (possible kick)
            if self.last_ball_direction > 0:
                direction_delta = abs(self.last_ball_direction - ball_direction)
                direction_delta = min(direction_delta, abs(360 - direction_delta))
                
                if direction_delta > 80 and ball_speed_kmh > 5:
                    # Possible sudden direction change
                    result['events'].append({
                        'type': 'direction_change',
                        'ball_direction': ball_direction,
                        'prev_direction': self.last_ball_direction,
                        'direction_delta': direction_delta
                    })
                    
                    # Reset kick detection on direction change
                    self.kick_detected = False
            
            # Check for kicks based on acceleration
            if ball_accel > 0:  # Only check valid acceleration values
                if ((ball_accel > 3 and not self.kick_detected) or 
                    (ball_accel > 6 and ball_speed_kmh < 26)):
                    # Kick detected
                    self.kick_detected = True
                    self.kick_ttl = 10  # Keep kick state for 10 frames
                    
                    # Add event
                    result['events'].append({
                        'type': 'kick',
                        'position': (x_2d, y_2d),
                        'speed': ball_speed_kmh,
                        'acceleration': ball_accel
                    })
                    
                    # Start new action
                    if not self.current_action:
                        self.current_action = {
                            'id': len(self.action_history) + 1,
                            'start_time': self.match_time_str,
                            'start_position': (x_2d, y_2d),
                            'ball_path': [(x_2d, y_2d)],
                            'speed': ball_speed_kmh,
                            'acceleration': ball_accel,
                            'type': 'Unknown'  # Will be classified later
                        }
            
            # Reset kick detection if ball slows down
            if ball_accel < 2 and ball_speed_kmh < 2:
                self.kick_detected = False
            
            # Decrease kick TTL
            if self.kick_ttl > 0:
                self.kick_ttl -= 1
                if self.kick_ttl == 0:
                    self.kick_detected = False
        
        # Update last valid positions
        self.last_valid_ball_x = center_x
        self.last_valid_ball_y = center_y
        self.last_ball_direction = ball_direction
        self.previous_ball_speed = ball_speed_kmh
        
        # Ball tracking data for this frame
        ball_data = {
            'center_x': center_x,
            'center_y': center_y,
            'x_2d': x_2d,
            'y_2d': y_2d,
            'radius': radius,
            'speed_kmh': ball_speed_kmh,
            'acceleration': ball_accel,
            'direction': ball_direction,
            'kick_detected': self.kick_detected
        }
        
        # Add to history
        self.ball_history.append(ball_data)
        
        # Update result
        result['ball_tracking'] = ball_data
        
        # Update current action if active
        if self.current_action:
            self.current_action['ball_path'].append((x_2d, y_2d))
            
            # Check if action is complete (ball stopped or changed direction)
            if self.kick_detected == False and len(self.current_action['ball_path']) > 5:
                # Classify action type based on distance
                start_x, start_y = self.current_action['start_position']
                end_x, end_y = (x_2d, y_2d)
                
                distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                
                if distance < 50:
                    action_type = "SHORT PASS"
                elif distance < 150:
                    action_type = "MEDIUM PASS"
                else:
                    action_type = "LONG PASS"
                
                # Check if it ended near goal
                if end_x < 50 and 100 < end_y < 150:
                    action_type = "SHOT AT LEFT GOAL"
                elif end_x > 450 and 100 < end_y < 150:
                    action_type = "SHOT AT RIGHT GOAL"
                
                self.current_action['type'] = action_type
                self.current_action['end_position'] = (x_2d, y_2d)
                self.current_action['end_time'] = self.match_time_str
                
                # Add to history and reset current
                self.action_history.append(self.current_action)
                if len(self.action_history) > self.max_actions:
                    self.action_history.pop(0)
                
                self.current_action = None
        
        return ball_data
    
    def process_player_detection(self, player_detection, result):
        """Process a player detection, tracking position and assigning IDs"""
        # Extract player data
        x1 = player_detection['x1']
        y1 = player_detection['y1']
        x2 = player_detection['x2']
        y2 = player_detection['y2']
        width = player_detection['width']
        height = player_detection['height']
        center_x = player_detection['center_x']
        center_y = player_detection['center_y']
        
        # Calculate 2D pitch coordinates (0-500 x, 0-250 y)
        x_2d = int(center_x / 640 * 500)  # Scale to pitch coords
        y_2d = int(center_y / 384 * 250)  # Scale to pitch coords
        
        # Simple tracking - match to existing player based on position
        player_id = self.assign_player_id(x_2d, y_2d, width, height)
        
        # Player tracking data for this frame
        player_data = {
            'player_id': player_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'center_x': center_x,
            'center_y': center_y,
            'x_2d': x_2d,
            'y_2d': y_2d,
            'width': width,
            'height': height,
            'team': 'unknown',  # Could be determined by jersey color
            'closest_to_ball': False
        }
        
        # Update player tracking
        self.player_tracking[player_id] = {
            'position': (x_2d, y_2d),
            'last_seen': self.frames_processed
        }
        
        # Update appearance count
        self.player_appearance_counts[player_id] += 1
        
        # Add to result
        result['player_tracking'].append(player_data)
        
        return player_data
    
    def assign_player_id(self, x_2d, y_2d, width, height):
        """Assign an ID to a player based on position tracking"""
        best_id = None
        best_distance = 50  # Maximum distance to consider same player
        
        # Look for matching existing player
        for player_id, data in self.player_tracking.items():
            # Skip if player hasn't been seen recently
            if self.frames_processed - data['last_seen'] > 30:
                continue
                
            px, py = data['position']
            distance = math.sqrt((x_2d - px) ** 2 + (y_2d - py) ** 2)
            
            if distance < best_distance:
                best_distance = distance
                best_id = player_id
        
        # If no match found, assign new ID
        if best_id is None:
            best_id = self.next_player_id
            self.next_player_id += 1
            self.player_tracking[best_id] = {
                'position': (x_2d, y_2d),
                'last_seen': self.frames_processed
            }
        
        return best_id
    
    def find_closest_player_to_ball(self, result):
        """Find the player closest to the ball"""
        ball_data = result['ball_tracking']
        if not ball_data:
            return
            
        ball_x = ball_data['x_2d']
        ball_y = ball_data['y_2d']
        
        closest_player = None
        closest_distance = float('inf')
        
        for player_data in result['player_tracking']:
            player_x = player_data['x_2d']
            player_y = player_data['y_2d']
            
            distance = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_player = player_data
        
        # Mark closest player and add to ball data
        if closest_player and closest_distance < 50:  # Only if reasonably close
            closest_player['closest_to_ball'] = True
            ball_data['closest_player_id'] = closest_player['player_id']
            
            # Add to events if kick detected
            if ball_data['kick_detected']:
                result['events'].append({
                    'type': 'player_kick',
                    'player_id': closest_player['player_id'],
                    'position': (ball_data['x_2d'], ball_data['y_2d'])
                })
    
    def check_for_goals(self, result):
        """Check if a goal has been scored"""
        if not result['ball_tracking']:
            return
            
        ball_data = result['ball_tracking']
        ball_x = ball_data['x_2d']
        ball_y = ball_data['y_2d']
        
        # Check left goal
        if self.goal_hold_back_left == -1:
            # Check if ball is in left goal area
            if (ball_x < self.goal_line_left_x + 5 and 
                self.goal_top_y < ball_y < self.goal_bottom_y):
                
                # Start goal verification countdown
                self.goal_hold_back_left = self.goal_hold_back_frames
                
                # Add goal event
                result['events'].append({
                    'type': 'potential_goal_left',
                    'position': (ball_x, ball_y),
                    'time': self.match_time_str
                })
        else:
            # Continue goal verification countdown
            self.goal_hold_back_left -= 1
            
            # If countdown reaches zero, confirm goal
            if self.goal_hold_back_left == 0:
                is_goal = self.verify_goal('left')
                
                if is_goal:
                    self.score_right += 1  # Right team scores in left goal
                    
                    # Add confirmed goal event
                    result['events'].append({
                        'type': 'goal_left',
                        'score_left': self.score_left,
                        'score_right': self.score_right,
                        'time': self.match_time_str
                    })
                    
                    # Update score in result
                    result['score']['left'] = self.score_left
                    result['score']['right'] = self.score_right
        
        # Check right goal (similar logic)
        if self.goal_hold_back_right == -1:
            # Check if ball is in right goal area
            if (ball_x > self.goal_line_right_x - 5 and 
                self.goal_top_y < ball_y < self.goal_bottom_y):
                
                # Start goal verification countdown
                self.goal_hold_back_right = self.goal_hold_back_frames
                
                # Add goal event
                result['events'].append({
                    'type': 'potential_goal_right',
                    'position': (ball_x, ball_y),
                    'time': self.match_time_str
                })
        else:
            # Continue goal verification countdown
            self.goal_hold_back_right -= 1
            
            # If countdown reaches zero, confirm goal
            if self.goal_hold_back_right == 0:
                is_goal = self.verify_goal('right')
                
                if is_goal:
                    self.score_left += 1  # Left team scores in right goal
                    
                    # Add confirmed goal event
                    result['events'].append({
                        'type': 'goal_right',
                        'score_left': self.score_left,
                        'score_right': self.score_right,
                        'time': self.match_time_str
                    })
                    
                    # Update score in result
                    result['score']['left'] = self.score_left
                    result['score']['right'] = self.score_right
    
    def verify_goal(self, side):
        """Verify if a goal was actually scored by analyzing ball trajectory"""
        # Look at ball history to confirm the goal
        # This is a simplified version - could be enhanced with more complex checks
        
        # Get the last few positions from history
        positions = []
        for ball_data in list(self.ball_history)[-self.goal_hold_back_frames:]:
            positions.append((ball_data['x_2d'], ball_data['y_2d']))
        
        # Check if enough positions are available
        if len(positions) < 5:
            return False
        
        # Check if ball was consistently in goal area
        goal_area_count = 0
        
        for x, y in positions:
            if side == 'left':
                in_goal_area = (x < self.goal_line_left_x + 10 and 
                               self.goal_top_y - 10 < y < self.goal_bottom_y + 10)
            else:  # right
                in_goal_area = (x > self.goal_line_right_x - 10 and 
                               self.goal_top_y - 10 < y < self.goal_bottom_y + 10)
            
            if in_goal_area:
                goal_area_count += 1
        
        # Confirm goal if ball was in goal area for enough frames
        return goal_area_count >= 3
    
    def update_possession_stats(self):
        """Update ball possession statistics"""
        if not self.ball_history:
            return
            
        last_ball_data = self.ball_history[-1]
        x_2d = last_ball_data['x_2d']
        
        # Simple possession based on which half the ball is in
        if x_2d < 250:  # Left half
            self.ball_possession['left'] += 1
        else:  # Right half
            self.ball_possession['right'] += 1
        
        # Calculate percentages
        total = self.ball_possession['left'] + self.ball_possession['right']
        if total > 0:
            left_percent = int(100 * self.ball_possession['left'] / total)
            right_percent = 100 - left_percent
            
            self.ball_possession['left_percent'] = left_percent
            self.ball_possession['right_percent'] = right_percent
    
    def get_match_stats(self):
        """Get current match statistics"""
        return {
            'score': {
                'left': self.score_left,
                'right': self.score_right
            },
            'possession': {
                'left': self.ball_possession.get('left_percent', 50),
                'right': self.ball_possession.get('right_percent', 50)
            },
            'match_time': self.match_time_str,
            'frames_processed': self.frames_processed,
            'ball_detected_frames': self.ball_detected_frames,
            'player_detected_frames': self.player_detected_frames,
            'player_count': len(self.player_tracking),
            'actions': len(self.action_history)
        }
    
    def calculate_moving_average(self, queue, new_value, default=-1):
        """Calculate moving average from a queue of values"""
        if new_value != default:
            queue.append(new_value)
        
        # If queue is empty or only has default values, return default
        valid_values = [v for v in queue if v != default]
        if not valid_values:
            return default
        
        # Calculate weighted average (newer values have higher weight)
        weights = list(range(1, len(valid_values) + 1))
        weighted_sum = sum(w * v for w, v in zip(weights, valid_values))
        total_weight = sum(weights)
        
        moving_average = weighted_sum / total_weight
        return round(moving_average, 2)