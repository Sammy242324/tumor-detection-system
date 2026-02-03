import os
from PIL import Image, ImageDraw, ImageFont

# Paths for the folders
base_path = "data/test"
categories = ["healthy", "normal", "tumor"]

# Create a blank image for each category
def create_image(category, filename):
    img = Image.new('RGB', (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Write text in the middle of the image
    draw.text((20, 50), category, fill=(0, 0, 0))
    
    # Save image
    img.save(filename)

# Generate images
for category in categories:
    folder_path = os.path.join(base_path, category)
    os.makedirs(folder_path, exist_ok=True)
    
    # Create 2 sample images per category
    for i in range(1, 3):
        file_path = os.path.join(folder_path, f"{category}_test{i}.jpg")
        create_image(category, file_path)

print("Sample test images generated successfully!")
