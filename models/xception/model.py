import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

def create_xception_model(input_shape=(299, 299, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
