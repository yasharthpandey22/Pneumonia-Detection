import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("pneumonia_model.keras")

print("Model Loaded Successfully")

# Change this to any image in your dataset
img_path = "chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize image
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

print("Prediction Value:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")