import numpy as np
import time
import os
import pickle

class BallRadiusTracker:
    def __init__(self, field_width=500, field_height=250, grid_width=25, grid_height=12, save_file='ball_radius_field_data.pkl'):
        self.field_width = field_width
        self.field_height = field_height
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.save_file = save_file
        self.last_save_time = time.time()  # Track the last time data was saved


        # Calculate cell size
        self.cell_width = field_width / grid_width
        self.cell_height = field_height / grid_height
        
        # Initialize a grid to store the sum of radii and count of entries for averaging
        self.radius_sum_grid = np.zeros((grid_width, grid_height))
        self.count_grid = np.zeros((grid_width, grid_height))

        self.load_data()     

    def update_radius(self, x, y, radius):
        # Determine which cell the (x, y) falls into
        grid_x = min(int(x // self.cell_width), self.grid_width - 1)
        grid_y = min(int(y // self.cell_height), self.grid_height - 1)

        # Update the sum and count grids
        self.radius_sum_grid[grid_x, grid_y] += radius
        self.count_grid[grid_x, grid_y] += 1

        self.check_save_data()

    def check_save_data(self):
        current_time = time.time()
        if current_time - self.last_save_time > 10:
            self.save_data()
            self.last_save_time = current_time

    def save_data(self):
        with open(self.save_file, 'wb') as f:
            pickle.dump((self.radius_sum_grid, self.count_grid), f)
        print("Data saved successfully.")

    def load_data(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'rb') as f:
                self.radius_sum_grid, self.count_grid = pickle.load(f)
            print("Data loaded successfully.")

    def get_grid_radius(self, x, y):
        grid_x = min(int(x // self.cell_width), self.grid_width - 1)
        grid_y = min(int(y // self.cell_height), self.grid_height - 1)

        return self.radius_sum_grid[grid_x, grid_y] / self.count_grid[grid_x, grid_y]

    def get_average_radius(self, x, y):
        # Determine which cell the (x, y) falls into
        grid_x = min(int(x // self.cell_width), self.grid_width - 1)
        grid_y = min(int(y // self.cell_height), self.grid_height - 1)
        
        # Check if the cell has at least 10 entries
        if self.count_grid[grid_x, grid_y] >= 10:
            res= self.radius_sum_grid[grid_x, grid_y] / self.count_grid[grid_x, grid_y]
            return res
        else:
            # Attempt to interpolate from neighboring cells
            neighbors = self.get_neighbors(grid_x, grid_y)
            avg_radii = [
                self.radius_sum_grid[nx, ny] / self.count_grid[nx, ny]
                for nx, ny in neighbors
                if self.count_grid[nx, ny] >= 10
            ]
            
            # Return the average of valid neighboring cells if available
            if avg_radii:
                return np.mean(avg_radii)
            else:
                # Return None if no sufficient data
                return None

    def get_neighbors(self, grid_x, grid_y):
        # Get the list of valid neighboring cells within the grid boundaries
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the current cell itself
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    neighbors.append((nx, ny))
        return neighbors

    def is_radius_valid(self, x, y, radius, tolerance):
        # Always update the grid with the new radius
        self.update_radius(x, y, radius)
        
        # Calculate the average radius at the specified (x, y)
        avg_radius = self.get_average_radius(x, y)

        if avg_radius is None:
            return True  # Return 0 if there is not enough data to calculate the difference

        if abs(avg_radius-radius)<=tolerance:
            return True
        
        return False
