import os
import cv2
import shutil

# Combined Label file
combined_label_file = "/media/Personal/NBV/Documents/USTH/Year_3/ComputerVision/traffic-signs-data/FullIJCNN2013/gt.txt"
# Image Directory
image_dir = "/media/Personal/NBV/Documents/USTH/Year_3/ComputerVision/images"
# Label Directory
output_label_dir = "/media/Personal/NBV/Documents/USTH/Year_3/ComputerVision/labels"

# Delete output label directory if exists
if os.path.exists(output_label_dir):
    shutil.rmtree(output_label_dir)

# Create output label directory if not exists
os.makedirs(output_label_dir, exist_ok=True)

# Dictionary to store annotations per image
annotations = {}

# Read the combined label file
with open(combined_label_file, "r") as file:
    for line in file:
        parts = line.strip().split(";")
        if len(parts) != 6:
            print(f"Warning: Skipping invalid line - {line.strip()}")
            continue
        
        img_name, left, top, right, bottom, class_id = parts
        left, top, right, bottom, class_id = map(int, [left, top, right, bottom, class_id])

        # Get image dimensions
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found, skipping...")
            continue
        
        img = cv2.imread(img_path)
        H, W, _ = img.shape  # Image height and width

        # Normalize bounding box values for YOLO format
        x_center = ((left + right) / 2) / W
        y_center = ((top + bottom) / 2) / H
        bbox_width = (right - left) / W
        bbox_height = (bottom - top) / H

        # Format annotation line in YOLO format
        annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

        # Remove ".ppm" extension for naming consistency
        img_name_no_ext = img_name.replace(".ppm", "")

        # Append annotation to the corresponding image key
        if img_name_no_ext not in annotations:
            annotations[img_name_no_ext] = []
        annotations[img_name_no_ext].append(annotation_line)

# Write each image's annotations to separate label files
for img_name, label_lines in annotations.items():
    label_file_path = os.path.join(output_label_dir, f"{img_name}.txt")
    
    with open(label_file_path, "w") as label_file:
        label_file.writelines(label_lines)

print("Label files have been successfully created in YOLO format!")