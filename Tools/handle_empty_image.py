import os

# Paths
image_dir = "/media/Personal/NBV/Documents/USTH/Year_3/ComputerVision/notebooks_area/images"  # Change to "val" if needed
label_dir = "/media/Personal/NBV/Documents/USTH/Year_3/ComputerVision/notebooks_area/labels"  # Change to "val" if needed
annotation_file = "/media/Personal/NBV/Documents/USTH/Year_3/ComputerVision/notebooks_area/gt.txt"  # Change to "gt_val.txt" for validation

# Get all image filenames (without extension)
image_files = {f.split(".")[0] for f in os.listdir(image_dir) if f.endswith(".ppm")}

# Get all images that have labels from annotation file
labeled_images = set()
with open(annotation_file, "r") as f:
    for line in f:
        img_name = line.split(";")[0]  # Extract image name
        img_name_no_ext = img_name.replace(".ppm", "")
        labeled_images.add(img_name_no_ext)

# Find images without labels
unlabeled_images = image_files - labeled_images

# Create empty label files for these images
for img_name in unlabeled_images:
    empty_label_path = os.path.join(label_dir, f"{img_name}.txt")
    open(empty_label_path, "w").close()  # Create an empty file

print(f"Created {len(unlabeled_images)} empty label files for images without annotations.")
