import cv2
import os
import csv
from tqdm import tqdm

def video_to_frames_and_crop(video_path, output_folder, csv_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read CSV annotations
    annotations = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame_number = int(row[list(row.keys())[0]])
            if frame_number not in annotations:
                annotations[frame_number] = []
            annotations[frame_number].append(row)

    frame_count = 1
    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in annotations:
                for annotation in annotations[frame_count]:
                    label = annotation['label']
                    coords = eval(annotation['coords'])  # Convert string to list

                    # Convert center-based coordinates to top-left-based coordinates
                    cx, cy, w, h = map(float, coords)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    w = int(w)
                    h = int(h)

                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    #mirror the box
                    y = frame.shape[0] - y - h
                    


                    # Create class folder if it doesn't exist
                    class_folder = os.path.join(output_folder, label)
                    if not os.path.exists(class_folder):
                        os.makedirs(class_folder)

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Crop the image
                    cropped_image = frame[y:y + h, x:x + w]

                    # Save the cropped image
                    cropped_filename = os.path.join(class_folder, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(cropped_filename, cropped_image)

            #Display the frame with bounding box
            cv2.imshow('Frame with Bounding Box', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames and saved cropped images to {output_folder}")


# Example usage
# video_path = 'dataset/SimData/SimData_Optical/SimData_2024-03-05__16-02-23.mp4'
# output_folder = 'dataset/SimData/SimData_Optical/output_frames'
# csv_path = 'dataset/SimData/SimData_Optical/SimData_2024-03-05__16-02-23.csv'
# video_path = "dataset/SimData/SimData_Depth/SimData_2024-03-05__16-28-05.mp4"
# output_folder = "dataset/SimData/SimData_Depth/output_frames"
# csv_path = "dataset/SimData/SimData_Depth/SimData_2024-03-05__16-28-05.csv"
video_path = "dataset/SimData/SimData_WhiteHot/SimData_2024-03-05__16-11-05.mp4"
output_folder = "dataset/SimData/SimData_WhiteHot/output_frames"
csv_path = "dataset/SimData/SimData_WhiteHot/SimData_2024-03-05__16-11-05.csv"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    print("Folder already exists")
    # Remove all files in the folder

video_to_frames_and_crop(video_path, output_folder, csv_path)