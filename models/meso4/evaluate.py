import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
model = tf.keras.models.load_model('models/meso4/meso4_model.h5')

# Data generator for testing
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/raw/test/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Evaluate the model
results = model.evaluate(test_generator)
print(f"Loss: {results[0]}")
print(f"Accuracy: {results[1]}")
