import numpy as np
import json
import os
from scipy.stats import linregress
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator

class Coordinate:
    def __init__(self, pt3dx, pt3dy, pt2dx, pt2dy, id=None):
        self.id = id
        self.pt_3d_x = pt3dx
        self.pt_3d_y = pt3dy
        self.pt_2d_x = pt2dx
        self.pt_2d_y = pt2dy
    
    def to_dict(self):
        return {"id": self.id, "3d_x": self.pt_3d_x, "3d_y": self.pt_3d_y, "2d_x": self.pt_2d_x, "2d_y": self.pt_2d_y}
    
    @staticmethod
    def from_dict(data):
        return Coordinate(data["3d_x"], data["3d_y"], data["2d_x"], data["2d_y"], data.get("id"))

class CoordinateManager:
    def __init__(self, filepath):
        self.coordinates = []
        self.filepath = filepath
        self.load_coordinates()
        
    def add_coordinate(self, coordinate):
        # Assign an ID to the coordinate based on the current number of coordinates
        coordinate.id = len(self.coordinates) + 1
        self.coordinates.append(coordinate)
        
    def load_coordinates(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as infile:
                data = json.load(infile)
                for item in data:
                    if item['2d_x'] > 0 and item['2d_y'] > 0: 
                        coord = Coordinate.from_dict(item)
                        # Ensure the ID is correctly set even after loading
                        if coord.id is None:
                            coord.id = len(self.coordinates) + 1
                        self.coordinates.append(coord)

            print(f'Loaded {len(self.coordinates)} reference points for 3D transformation')
        else:
            print("Camera mapping file not found - filename:", self.filepath)
    
    def save_coordinates(self):
        with open(self.filepath, "w") as outfile:
            # Ensure coordinates are sorted by ID before saving
            sorted_coords = sorted(self.coordinates, key=lambda x: x.id if x.id is not None else 0)
            # Use indent to format entries with newlines and indentation
            json.dump([coord.to_dict() for coord in sorted_coords], outfile, indent=4)


class CameraMapper:
    def __init__(self, filename):
        self.filename = filename 
        self.manager = CoordinateManager(filename)
        self.rbf_function = 'linear'
        self.epsilon = 1
        self.f_2d_x = None
        self.f_2d_y = None

    def prepare_interpolation(self):
        # Extracting 3D and 2D coordinates
        points_3d = np.array([(coord.pt_3d_x, coord.pt_3d_y) for coord in self.manager.coordinates])
        values_2d_x = np.array([coord.pt_2d_x for coord in self.manager.coordinates])
        values_2d_y = np.array([coord.pt_2d_y for coord in self.manager.coordinates])
        
        self.f_2d_x = LinearNDInterpolator(points_3d, values_2d_x)
        self.f_2d_y = LinearNDInterpolator(points_3d, values_2d_y)
        

    def map_3d_to_2d(self, x, y):
        try:
            if self.f_2d_x is None:
                self.prepare_interpolation()
            
            x_2d = self.f_2d_x(x, y)
            y_2d = self.f_2d_y(x, y)

            x_2d = int(x_2d)
            y_2d = int(y_2d)
            
            if x_2d is None or y_2d is None:
                print("Interpolation resulted in None value, indicating that the point is outside the convex hull of the input data.")
                return -1, -1
        except:
            return -1, -1
        
        return x_2d, y_2d
    
    def estimate_3d_coordinate(self, pt_2d_x, pt_2d_y, rbf_function='multiquadric', epsilon=1):
        try:
            # Ensure there are enough points for interpolation
            if len(self.coordinates) < 3:
                print("Not enough points for interpolation. Need at least 3.")
                return 0, 0  # Returning default value

            # Extracting 3D and 2D coordinates
            _3d_xs = np.array([coord.pt_3d_x for coord in self.manager.coordinates])
            _3d_ys = np.array([coord.pt_3d_y for coord in self.manager.coordinates])
            _2d_xs = np.array([coord.pt_2d_x for coord in self.manager.coordinates])
            _2d_ys = np.array([coord.pt_2d_y for coord in self.manager.coordinates])

            # Create RBF interpolators with the specified function and epsilon
            rbf_3d_x = Rbf(_2d_xs, _2d_ys, _3d_xs, function=rbf_function, epsilon=epsilon)
            rbf_3d_y = Rbf(_2d_xs, _2d_ys, _3d_ys, function=rbf_function, epsilon=epsilon)

            # Estimate 3D coordinates using RBF interpolation
            estimated_3d_x = rbf_3d_x(pt_2d_x, pt_2d_y)
            estimated_3d_y = rbf_3d_y(pt_2d_x, pt_2d_y)

            return estimated_3d_x, estimated_3d_y
        except Exception as e:
            print(f"An error occurred: {e}")
            return -1, -1
        
    def estimate_2d_coordinate(self, pt_3d_x, pt_3d_y):
        try:
            return self.map_3d_to_2d(pt_3d_x, pt_3d_y)
        except:
            return -1,-1
        try:
            # Ensure there are enough points for interpolation
            if len(self.manager.coordinates) < 3:
                print("Not enough points for interpolation. Need at least 3.")
                return 0, 0  # Returning default value

            # Extracting 3D and 2D coordinates
            _3d_xs = np.array([coord.pt_3d_x for coord in self.manager.coordinates])
            _3d_ys = np.array([coord.pt_3d_y for coord in self.manager.coordinates])
            _2d_xs = np.array([coord.pt_2d_x for coord in self.manager.coordinates])
            _2d_ys = np.array([coord.pt_2d_y for coord in self.manager.coordinates])

            # Create RBF interpolators with the specified function and epsilon
            rbf_x = Rbf(_3d_xs, _3d_ys, _2d_xs, function=self.rbf_function, epsilon=self.epsilon)
            rbf_y = Rbf(_3d_xs, _3d_ys, _2d_ys, function=self.rbf_function, epsilon=self.epsilon)

            # Estimate 2D coordinates using RBF interpolation
            estimated_2d_x = rbf_x(pt_3d_x, pt_3d_y)
            estimated_2d_y = rbf_y(pt_3d_x, pt_3d_y)

            return estimated_2d_x, estimated_2d_y
        except:
            return -1,-1