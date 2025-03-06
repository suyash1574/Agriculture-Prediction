import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("models/crop_model.h5")
CROP_TYPES = ["Tomato", "Corn", "Potato", "Apple"]
FERTILIZATION_CLASSES = ["Low", "Medium", "High"]

# Load min/max from training (hardcoded for now; ideally save these during training)
YIELD_MIN = 2463  # Update with actual min from your dataset
YIELD_MAX = 385434  # Update with actual max from your dataset

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    tabular_input_shape = model.input[1].shape[1]
    tabular_data = np.random.rand(1, tabular_input_shape)
    
    predictions = model.predict([img, tabular_data])
    
    crop_type = CROP_TYPES[np.argmax(predictions[0][0])]
    destruction_percentage = predictions[1][0][0] * 100
    yield_prediction_normalized = predictions[2][0][0]  # Normalized [0, 1]
    yield_prediction = yield_prediction_normalized * (YIELD_MAX - YIELD_MIN) + YIELD_MIN  # Denormalize to hg/ha
    yield_prediction_tons_ha = yield_prediction / 10000  # Convert to tons/ha
    fertilization = FERTILIZATION_CLASSES[np.argmax(predictions[3][0])]
    
    return {
        "crop_type": crop_type,
        "destruction_percentage": round(destruction_percentage, 2),
        "yield_prediction": round(yield_prediction_tons_ha, 2),
        "fertilization": fertilization
    }