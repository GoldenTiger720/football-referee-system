import cv2
import numpy as np
from PIL import Image, ImageTk

def resize_image_to_fit(image, max_width, max_height):
    """
    Resize an image to fit within the specified dimensions while preserving aspect ratio
    
    Args:
        image: PIL Image or OpenCV image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    # Convert OpenCV image to PIL if needed
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image
    
    # Get original dimensions
    width, height = pil_image.size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Calculate new dimensions
    if width > height:
        # Landscape
        new_width = min(width, max_width)
        new_height = int(new_width / aspect_ratio)
        
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
    else:
        # Portrait or square
        new_height = min(height, max_height)
        new_width = int(new_height * aspect_ratio)
        
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
    
    # Resize image
    resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert back to OpenCV if input was OpenCV
    if isinstance(image, np.ndarray):
        resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    
    return resized_image

def convert_cv2_to_tkinter(cv2_image):
    """
    Convert an OpenCV image to a format suitable for Tkinter
    
    Args:
        cv2_image: OpenCV image (BGR format)
        
    Returns:
        Tkinter-compatible PhotoImage
    """
    # Convert color format: BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Convert to PhotoImage
    tk_image = ImageTk.PhotoImage(image=pil_image)
    
    return tk_image

def draw_detections(frame, detections, draw_scores=True):
    """
    Draw bounding boxes, labels, and scores on an image
    
    Args:
        frame: OpenCV image to draw on
        detections: List of detection results
        draw_scores: Whether to draw confidence scores
        
    Returns:
        Annotated image
    """
    # Make a copy to avoid modifying the original
    annotated = frame.copy()
    
    for det in detections:
        # Extract data
        cls = det.get('class', 'unknown')
        x1 = int(det.get('x1', 0))
        y1 = int(det.get('y1', 0))
        x2 = int(det.get('x2', 0))
        y2 = int(det.get('y2', 0))
        conf = det.get('confidence', 0.0)
        
        # Determine color based on class
        if cls == 'ball':
            color = (0, 0, 255)  # Red (BGR)
        elif cls == 'fastball':
            color = (0, 165, 255)  # Orange
        elif cls == 'player':
            color = (0, 255, 0)  # Green
        else:
            color = (255, 255, 255)  # White
        
        # Draw bounding box
        if cls in ['ball', 'fastball']:
            # For balls, draw a circle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min(x2 - x1, y2 - y1) // 2
            
            cv2.circle(annotated, (center_x, center_y), radius, color, 2)
            
            # Draw a smaller filled circle at the center
            cv2.circle(annotated, (center_x, center_y), 2, color, -1)
        else:
            # For other objects, draw a rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        if draw_scores:
            label = f"{cls} {conf:.2f}"
        else:
            label = cls
        
        # Get text size for proper background sizing
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw text background
        cv2.rectangle(
            annotated,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            2
        )
    
    return annotated

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: (x, y) tuple
        point2: (x, y) tuple
        
    Returns:
        Distance between points
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2) tuple
        box2: (x1, y1, x2, y2) tuple
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    # Convert to arrays for easier calculation
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes overlap
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate area of intersection
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def calculate_moving_average(queue, new_value, default_value=-1):
    """
    Calculate moving average of values in a queue
    
    Args:
        queue: deque object with values
        new_value: New value to add to queue
        default_value: Default value to return if no valid values
        
    Returns:
        Moving average of queue values
    """
    # Add new value if not default
    if new_value != default_value:
        queue.append(new_value)
    
    # Get valid values (not default)
    valid_values = [v for v in queue if v != default_value]
    
    # Return default if no valid values
    if not valid_values:
        return default_value
    
    # Calculate weighted average (newer values have higher weight)
    weights = list(range(1, len(valid_values) + 1))
    weighted_sum = sum(w * v for w, v in zip(weights, valid_values))
    total_weight = sum(weights)
    
    moving_average = weighted_sum / total_weight
    return round(moving_average, 2)

def scale_coordinates(coords, source_size, target_size):
    """
    Scale coordinates from one size to another
    
    Args:
        coords: (x, y) coordinates
        source_size: (width, height) of source
        target_size: (width, height) of target
        
    Returns:
        Scaled coordinates (x, y)
    """
    x, y = coords
    src_width, src_height = source_size
    tgt_width, tgt_height = target_size
    
    # Scale coordinates
    scaled_x = int(x * tgt_width / src_width)
    scaled_y = int(y * tgt_height / src_height)
    
    return scaled_x, scaled_y