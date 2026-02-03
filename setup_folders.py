import os

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')


classes = ['tumor', 'healthy']


for directory in [train_dir, val_dir]:
    for class_name in classes:
        path = os.path.join(directory, class_name)
        os.makedirs(path, exist_ok=True)

print(" Folder structure created:")
for directory in [train_dir, val_dir]:
    for class_name in classes:
        print(f"- {os.path.join(directory, class_name)}")
