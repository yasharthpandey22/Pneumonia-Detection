import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "chest_xray/train"
test_path = "chest_xray/test"
train_datagen = ImageDataGenerator(
    rescale=1.0/255
)
test_datagen = ImageDataGenerator(
    rescale=1.0/255
)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)
print("Training Images:", train_generator.samples)
print("Testing Images:", test_generator.samples)

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(224,224,3)
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)
model.save("pneumonia_model.keras")

print("Model Saved Successfully!")