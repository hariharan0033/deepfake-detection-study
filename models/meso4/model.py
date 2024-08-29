import tensorflow as tf
from tensorflow.keras import layers, models

def create_meso4_model(input_shape=(256, 256, 3)):
    model = models.Sequential()
    
    # Add layers according to Meso4 architecture
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
