from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

# Define paths
raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'

# Create directories if they don't exist
os.makedirs(processed_data_path + 'train/real', exist_ok=True)
os.makedirs(processed_data_path + 'train/fake', exist_ok=True)
os.makedirs(processed_data_path + 'validation/real', exist_ok=True)
os.makedirs(processed_data_path + 'validation/fake', exist_ok=True)
os.makedirs(processed_data_path + 'test/real', exist_ok=True)
os.makedirs(processed_data_path + 'test/fake', exist_ok=True)

# Copy and preprocess data (e.g., resizing, normalization)
def preprocess_and_save(src_dir, dest_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        src_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary',
        save_to_dir=dest_dir,
        save_format='jpeg'
    )
    for _ in range(generator.samples // 32):
        generator.next()

preprocess_and_save(raw_data_path + 'train/', processed_data_path + 'train/')
preprocess_and_save(raw_data_path + 'validation/', processed_data_path + 'validation/')
preprocess_and_save(raw_data_path + 'test/', processed_data_path + 'test/')
