import cv2
import math
from enum import IntEnum
import numpy as np
from collections import defaultdict
import os
import scipy.cluster
import sklearn.cluster
from collections import Counter

MAX_COLORS = 12

import time


class ProcTimer:
    def __init__(self, debug=True):
        """Initializes the timer and starts it immediately."""
        self.debug = debug
        self.start()

    def start(self):
        if self.debug == False:
            return
        self.start_time = time.perf_counter_ns()
        self.end_time = None

    def stop(self, message=""):
        if self.debug == False:
            return
        """Stops the timer, calculates the elapsed time, and logs it with a message."""
        self.end_time = time.perf_counter_ns()
        elapsed_time_ns = self.end_time - self.start_time
        elapsed_time_ms = elapsed_time_ns / 1_000_000  # Convert to milliseconds
        print(f"[{message}] {elapsed_time_ms:.1f} ms")


class HumanFeatures:
    def __init__(self):
        self.shirt_color1 = None
        self.pants_color1 = None
        self.socks_color1 = None
        self.shirt_color2 = None
        self.pants_color2 = None
        self.socks_color2 = None

    def print(self):
        print(f"SHIRT: ", self.shirt_color1, self.shirt_color2)
        print(f"PANTS: ", self.pants_color1, self.pants_color2)


class ImageObject:
    def __init__(self, img):
        # dimensions
        self.original_width = -1
        self.original_height = -1
        self.resized_width = -1
        self.resized_height = -1
        self.tile_width = -1
        self.tile_height = -1

        # images
        self.image = img
        self.resized = None
        self.segment = None
        self.segment_w_points = None
        self.segment_w_poly = None
        self.mask = None

        self.filename = None
        self.aspect_ratio = -1
        self.grid_col = -1
        self.grid_row = -1
        self.area_ratio = -1
        self.total_confidence = None
        self.avrg_confidence = -1

        self.keypoints = None
        self.valid = False
        self.features = HumanFeatures()
        self.playerObj = None

    def set_grid_position(self, cols, rows):
        self.grid_col = cols
        self.grid_row = rows

    def reset(self):
        self.original_width = -1
        self.original_height = -1
        self.resized_width = -1
        self.resized_height = -1
        self.tile_width = -1
        self.tile_height = -1

        self.resized = None
        self.segment = None
        self.segment_w_points = None
        self.segment_w_poly = None
        self.mask = None

        self.aspect_ratio = -1
        self.grid_col = -1
        self.grid_row = -1
        self.area_ratio = -1
        self.total_confidence = None
        self.avrg_confidence = -1

        self.keypoints = None
        self.valid = False
        self.features = HumanFeatures()


class BodyPoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class PlayerUtils:
    def __init__(self):
        pass
    @staticmethod
    def stitch_images(image_objects, canvas_size=640, debug=False):
        if not image_objects:
            return None, None, None

        num_images = len(image_objects)

        # Calculate the grid size
        grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)

        if grid_cols < 2:
            grid_cols = 2
        if grid_rows < 2:
            grid_rows = 2

        # Calculate the maximum size of each cell in the grid
        max_cell_width = canvas_size // grid_cols
        max_cell_height = canvas_size // grid_rows

        def adjust_image(image_obj):
            # image_obj.image=adjust_white_balance(image_obj.image)

            # image_obj.image = cv2.fastNlMeansDenoisingColored(image_obj.image,None,10,10,7,21)
            # image_obj.image = cv2.fastNlMeansDenoisingMulti(image_obj.image, 2, 5, None, 4, 7, 35)

            # cv2.imwrite("sgarp.png", dst)

            return image_obj

        def adjust_white_balance(image, aggression_factor=1.5):
            """Adjust the white balance of an image more aggressively."""
            # Convert image to float32 for precise calculations
            image_float = image.astype(np.float32)

            # Calculate mean of each color channel
            mean_b = np.mean(image_float[:, :, 0])
            mean_g = np.mean(image_float[:, :, 1])
            mean_r = np.mean(image_float[:, :, 2])

            # Calculate the mean gray value
            mean_gray = (mean_b + mean_g + mean_r) / 3

            # Compute scaling factors for each channel with aggression factor
            scale_b = (mean_gray / mean_b) ** aggression_factor
            scale_g = (mean_gray / mean_g) ** aggression_factor
            scale_r = (mean_gray / mean_r) ** aggression_factor

            # Apply scaling to each channel
            image_float[:, :, 0] *= scale_b
            image_float[:, :, 1] *= scale_g
            image_float[:, :, 2] *= scale_r

            # Clip the values to the valid range and convert back to uint8
            image_balanced = np.clip(image_float, 0, 255).astype(np.uint8)

            return image_balanced

        def resize_with_aspect_ratio(image_obj, max_width, max_height, debug=False):
            """Resize the image to fill the cell while maintaining aspect ratio."""
            image_obj.original_height, image_obj.original_width = image_obj.image.shape[:2]
            image_obj.aspect_ratio = image_obj.original_width / image_obj.original_height
            new_width, new_height = 95, 186
            # if image_obj.aspect_ratio > 1:  # Wider than tall
            #     new_width = max_width
            #     new_height = int(new_width / image_obj.aspect_ratio)
            #     if new_height > max_height:
            #         new_height = max_height
            #         new_width = int(new_height * image_obj.aspect_ratio)
            # else:  # Taller than wide or square
            #     new_height = max_height
            #     new_width = int(new_height * image_obj.aspect_ratio)
            #     if new_width > max_width:
            #         new_width = max_width
            #         new_height = int(new_width / image_obj.aspect_ratio)

            # Resize the image to the new dimensions
            image_obj.resized = cv2.resize(image_obj.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            image_obj.resized_height = new_height
            image_obj.resized_width = new_width
            if debug:
                print(
                    f'Image {image_obj.filename} resized from {image_obj.original_width}x{image_obj.original_height} to {image_obj.resized_width}x{image_obj.resized_height}')
            return image_obj

        # Adjust white balance and resize images
        processed_image_objects = [
            resize_with_aspect_ratio(adjust_image(image_obj), max_cell_width, max_cell_height, debug) for image_obj in
            image_objects]
        # resize_with_aspect_ratio(image_obj, max_cell_width, max_cell_height, debug) for image_obj in image_objects]

        # Create a blank canvas to place the images
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        for idx, img_obj in enumerate(processed_image_objects):
            img_obj.grid_row = idx // grid_cols
            img_obj.grid_col = idx % grid_cols

            start_y = img_obj.grid_row * max_cell_height
            start_x = img_obj.grid_col * max_cell_width

            img_obj.resized_height, img_obj.resized_width = img_obj.resized.shape[:2]

            # Calculate the offset to center the image within its cell
            offset_y = (max_cell_height - img_obj.resized_height) // 2
            offset_x = (max_cell_width - img_obj.resized_width) // 2

            # Place the image on the canvas
            canvas[start_y + offset_y:start_y + offset_y + img_obj.resized_height,
            start_x + offset_x:start_x + offset_x + img_obj.resized_width] = img_obj.resized

        return grid_cols, grid_rows, canvas

    @staticmethod
    def rgb_diff(c1, c2):
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        return max(abs(r1 - r2), abs(g1 - g2), abs(b1 - b2))

    @staticmethod
    def color_avrg(c1, c2):
        r1, g1, b1 = c1
        r2, g2, b2 = c2

        return (int((r1 + r2) / 2), int((g1 + g2) / 2), int((b1 + b2) / 2))

    @staticmethod
    #    def euclidean_distance(color1, color2):
    #        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    def euclidean_distance(color1, color2):
        return math.sqrt(sum((abs(int(c1) - int(c2))) ** 2 for c1, c2 in zip(color1, color2)))

    @staticmethod
    def average_color(group):
        r_avg = int(sum(color[0] for color in group) / len(group))
        g_avg = int(sum(color[1] for color in group) / len(group))
        b_avg = int(sum(color[2] for color in group) / len(group))
        return (r_avg, g_avg, b_avg)

    @staticmethod
    def group_colors(colors, tolerance=20):
        groups = []
        for color in colors:
            found_group = False
            for group in groups:
                if PlayerUtils.euclidean_distance(color, group[0]) < tolerance:
                    group.append(color)
                    found_group = True
                    break
            if not found_group:
                groups.append([color])
        return groups

    @staticmethod
    def most_common_colors(colors, tolerance=20, top_n=2):
        grouped_colors = PlayerUtils.group_colors(colors, tolerance)
        color_averages = [(PlayerUtils.average_color(group), len(group)) for group in grouped_colors]
        color_averages.sort(key=lambda x: x[1], reverse=True)  # Sort by count in descending order
        return color_averages[:top_n]

    @staticmethod
    def find_dominant_color(image, mask):
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (0, 0, 0), (0, 0, 0)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        rect_image = masked_image[y:y + h, x:x + w]

        resized_image = cv2.resize(rect_image, (5, 5), interpolation=cv2.INTER_LINEAR)

        pixels = resized_image.reshape(-1, 3)

        valid_pixels = [tuple(pixel) for pixel in pixels if np.any(pixel > 0)]
        if not valid_pixels:
            return (0, 0, 0), (0, 0, 0)

        color_counts = Counter(valid_pixels)

        most_common_colors = PlayerUtils.most_common_colors(valid_pixels)  # color_counts.most_common(2)

        (r0, g0, b0), c0 = most_common_colors[0]

        if len(most_common_colors) > 1:
            (r1, g1, b1), c1 = most_common_colors[1]
        else:
            r1, g1, b1, c1 = r0, g0, b0, c0

        if c0 > c1 * 1.5:
            r1, g1, b1 = r0, g0, b0

        return (int(r0), int(g0), int(b0)), (int(r1), int(g1), int(b1))

    @staticmethod
    def average_color_in_polygon(img_obj, points, debug=False):
        # porcess_time = ProcTimer()
        # Define polygon from points
        polygon = np.array(points, dtype=np.int32)

        # Create mask
        mask = np.zeros(img_obj.segment.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Find dominant color in the masked area
        dominant_color = PlayerUtils.find_dominant_color(img_obj.segment, mask)

        # Optional debugging: draw polygon on image
        if debug:
            cv2.polylines(img_obj.segment_w_poly, [polygon], True, (0, 255, 0), 1)
        # porcess_time.stop("   Dominant color")
        return dominant_color

    @staticmethod
    def display_color(img, width, y, text, color1, color2):
        if color1 is not None:
            b, g, r = color1
            cv2.putText(img, text, (width - 190, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 100], thickness=1)
            cv2.putText(img, f'{r},{g},{b}', (width - 130, y + 15), cv2.FONT_HERSHEY_PLAIN, 0.8, [0, 255, 255],
                        thickness=1)

            cv2.rectangle(img, (width - 40, y), (width - 40 + 20, y + 10), color1, -1)
            cv2.rectangle(img, (width - 40, y + 10), (width - 40 + 20, y + 20), color2, -1)
            cv2.rectangle(img, (width - 40, y), (width - 40 + 20, y + 20), (255, 255, 255), 1)

    @staticmethod
    def get_filename_without_extension(file_path):
        base_name = os.path.basename(file_path)
        filename, _ = os.path.splitext(base_name)
        return filename

    @staticmethod
    def save_debug_image(folder, img_obj):
        if img_obj is None or img_obj.image is None:
            return
        fname_without_ext = PlayerUtils.get_filename_without_extension(img_obj.filename)
        # cv2.imwrite(os.path.join(folder, f"{img_obj.grid_col}_{img_obj.grid_row}.png"),img_obj.image)
        cv2.imwrite(os.path.join(folder, f"{fname_without_ext}.png"), img_obj.image)
        if img_obj.valid == False:
            return

        _, img_w = img_obj.resized.shape[:2]
        img_h = img_obj.resized_height
        new_width = img_w * 5 + 200

        new_image = np.zeros((int(img_h), new_width, 3), dtype=np.uint8)

        seg_h, seg_w = img_obj.resized.shape[:2]
        new_image[0:seg_h, 0:seg_w] = img_obj.resized
        # cv2.imwrite(f'debug_image_in_img_obj_resized{time.time()}.jpg',img_obj.resized)
        # cv2.imwrite(f'debug_image_in_img_obj_new_image{time.time()}.jpg',new_image[0:seg_h, 0:seg_w])
        if img_obj.mask is not None:
            seg_h, seg_w = img_obj.mask.shape[:2]
            if seg_h > 0 and seg_w > 0:
                new_image[0:seg_h, img_w * 1:img_w * 1 + seg_w] = cv2.merge([img_obj.mask, img_obj.mask, img_obj.mask])
                # cv2.imwrite(f'debug_image_in_img_obj_mask{time.time()}.jpg',new_image[0:seg_h, img_w * 1:img_w * 1 + seg_w])

        if img_obj.segment is not None:
            seg_h, seg_w = img_obj.segment.shape[:2]
            if seg_h > 0 and seg_w > 0:
                new_image[0:seg_h, img_w * 2:img_w * 2 + seg_w] = img_obj.segment
                # cv2.imwrite(f'debug_image_in_img_obj_segment{time.time()}.jpg',new_image[0:seg_h, img_w * 2:img_w * 2 + seg_w])
                if img_obj.keypoints is not None:
                    for x, y, conf in img_obj.keypoints:
                        if (conf > 0.8):
                            cv2.circle(img_obj.segment_w_points, (int(x), int(y)), 3, (0, 255, 0), -1)
                            # cv2.circle(img_obj.image, (int(x), int(y)), 3, (0, 255, 0), -1)
                        else:
                            cv2.circle(img_obj.segment_w_points, (int(x), int(y)), 3, (0, 0, 255), -1)
                            # cv2.circle(img_obj.image, (int(x), int(y)), 3, (0, 0, 255), -1)

                new_image[0:seg_h, img_w * 3:img_w * 3 + seg_w] = img_obj.segment_w_points
                try:
                    new_image[0:seg_h, img_w * 4:img_w * 4 + seg_w] = img_obj.segment_w_poly
                except:
                    pass

        # print("Shirt:", img_obj.features.shirt_color)
        # PlayerUtils.display_color(new_image, new_width, 10, "SHIRT:", img_obj.features.shirt_color1,
        #                           img_obj.features.shirt_color2)

        # PlayerUtils.display_color(new_image, new_width, 60, "PANTS:", img_obj.features.pants_color1,
        #                           img_obj.features.pants_color2)
        # PlayerUtils.display_color(new_image, new_width, 110, "SOCKS:", img_obj.features.socks_color1,
        #                           img_obj.features.socks_color2)
        PlayerUtils.display_color(new_image, new_width, 10, "SHIRT:", img_obj.features.shirt_color1,
                                  img_obj.features.shirt_color2)

        PlayerUtils.display_color(new_image, new_width, 60, "PANTS:", img_obj.features.pants_color1,
                                  img_obj.features.pants_color2)
        PlayerUtils.display_color(new_image, new_width, 110, "SOCKS:", img_obj.features.socks_color1,
                                  img_obj.features.socks_color2)
        
        cv2.putText(new_image, f'CONF:{img_obj.avrg_confidence}', (new_width - 100, 150), cv2.FONT_HERSHEY_PLAIN, 1,
                    [0, 255, 100], thickness=1)

        # cv2.imwrite(os.path.join(folder, f"{img_obj.grid_col}_{img_obj.grid_row}_processed.png"),new_image)
        cv2.imwrite(os.path.join(folder, f"{fname_without_ext}_processed.png"), new_image)
