from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Set confidence threshold
conf_threshold = 0.2

def process_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the color space from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save frame temporarily to process
        frame_path = f'temp_frame_{frame_count}.jpg'
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Run batched inference on the frame
        results = model([frame_path])  # Assuming model processes images in batches

        for result in results:
            if result.boxes is not None and hasattr(result.boxes, 'xyxy') and result.boxes.xyxy[0].numel() > 0:
                boxes = result.boxes.xyxy[0]
                # Ensure we have the correct dimensions before attempting to index
                if boxes.dim() == 2 and boxes.size(1) >= 5:
                    confidences = boxes[:, 4]  # Confidence scores are at index 4
                    valid_indices = confidences > conf_threshold
                    valid_boxes = boxes[valid_indices]

                    # Draw bounding boxes on the original frame
                    for box in valid_boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                else:
                    print("No valid boxes detected or incorrect dimensions.")
            else:
                print("No boxes available or empty detection.")

            # Save processed frame to disk
            output_path = f'output_frame_{frame_count}.jpg'
            cv2.imwrite(output_path, frame)

        frame_count += 1

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")


video_path = 'c:\Develop\cam3.mp4'
process_video(video_path)
