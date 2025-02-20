import os
from PIL import Image

# Define paths
image_folder = "D:/USTH/Year_3/Computer_Vision/Project/Traffic-Signs-Detection/Data/YoLo/Test"
output_folder = "D:/USTH/Year_3/Computer_Vision/Project/Traffic-Signs-Detection/Data/YoLo/Test/images"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Convert PPM to JPG
for filename in os.listdir(image_folder):
    if filename.endswith(".ppm"):
        ppm_path = os.path.join(image_folder, filename)
        img = Image.open(ppm_path)
        
        # Save as JPG
        jpg_filename = filename.replace(".ppm", ".jpg")
        img.save(os.path.join(output_folder, jpg_filename), "JPEG")

print("✅ Conversion complete: PPM -> JPG")
