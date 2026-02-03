import os
import shutil
import random


source_dir = 'all_images'  
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

classes = ['tumor', 'healthy']

val_split = 0.2


for directory in [train_dir, val_dir]:
    for class_name in classes:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)


for class_name in classes:
    class_source = os.path.join(source_dir, class_name)
    images = os.listdir(class_source)
    random.shuffle(images)
    val_count = int(len(images) * val_split)
    
    val_images = images[:val_count]
    train_images = images[val_count:]
    
    print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val")
    
    for img in train_images:
        src_path = os.path.join(class_source, img)
        dst_path = os.path.join(train_dir, class_name, img)
        shutil.copy2(src_path, dst_path)
    
    for img in val_images:
        src_path = os.path.join(class_source, img)
        dst_path = os.path.join(val_dir, class_name, img)
        shutil.copy2(src_path, dst_path)

print(" Dataset split complete.")
