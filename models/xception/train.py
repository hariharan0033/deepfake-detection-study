import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_xception_model

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=20, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True, 
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/raw/train/',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/raw/validation/',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

# Model training
model = create_xception_model()
history = model.fit(train_generator, 
                    epochs=10, 
                    validation_data=validation_generator)

# Save the model
model.save('models/xception/xception_model.h5')
