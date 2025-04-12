
def insert_for_even_distribution(processed_frames, available_frames):
    if len(processed_frames)==0:
        return available_frames[0]
    if len(processed_frames)==1:
        return available_frames[len(available_frames)-1]

    # Sort frames to ensure they are in ascending order
    processed_frames.sort()

    # List to hold gaps and their starting index (gap size, start index)
    gaps = []

    # Calculate gaps and store them with their start index
    for i in range(len(processed_frames) - 1):
        current_gap = processed_frames[i+1] - processed_frames[i]
        # Store the gap size and the index before the gap starts
        gaps.append((current_gap, i))

    # Sort gaps by size in descending order
    gaps.sort(reverse=True, key=lambda x: x[0])

    # Try to fill gaps from the largest to the smallest
    for gap_size, start_index in gaps:
        # Calculate the target frame for the current gap
        target_frame = (processed_frames[start_index+1] + processed_frames[start_index]) // 2

        # Initialize best fit for the current gap
        best_fit = None
        for frame in available_frames:
            if processed_frames[start_index] < frame < processed_frames[start_index+1]:
                if best_fit is None or abs(frame - target_frame) < abs(best_fit - target_frame):
                    best_fit = frame

        # If a suitable frame is found for this gap, insert it and return
        if best_fit is not None:
            # Here you might want to actually insert the frame into processed_frames if needed
            # processed_frames.insert(start_index + 1, best_fit)
            return best_fit

    # If the code reaches this point, it means no suitable frame was found for any gap
    return None
