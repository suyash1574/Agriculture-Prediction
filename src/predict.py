# src/predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
try:
    model = load_model('models/crop_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Load the scaler
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load scaler: {str(e)}")
    scaler = None

crop_types = ['Wheat', 'Corn', 'Rice', 'Soybean']  # Adjust based on your dataset

# Min/max values for denormalization (update after running train_model.py)
YIELD_MIN = 0.0  # Temporary, replace with actual min from training logs
YIELD_MAX = 1.0  # Temporary, replace with actual max from training logs
FERTILIZATION_MIN = 0.0
FERTILIZATION_MAX = 100.0

def preprocess_image(image_path, rainfall_value=0.0):
    """
    Preprocess the image and tabular data (Rainfall) for prediction.
    """
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prepare tabular data with the provided rainfall value
        tabular_data = np.array([[rainfall_value]], dtype=np.float32)
        if scaler is not None:
            tabular_data = scaler.transform(tabular_data)
        else:
            logger.warning("Scaler not loaded, using raw rainfall value")
        logger.info(f"Processed tabular data: {tabular_data}")

        return img_array, tabular_data
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, None

def predict_crop(image_path, rainfall_value=0.0):
    """
    Predict crop attributes using the model.
    """
    if model is None:
        return {'error': 'Model not loaded'}

    img_array, tabular_data = preprocess_image(image_path, rainfall_value)
    if img_array is None:
        return {'error': 'Failed to preprocess image'}

    try:
        # Make prediction
        predictions = model.predict([img_array, tabular_data])
        logger.info(f"Raw predictions: {predictions}")

        # Process predictions
        crop_type_pred = np.argmax(predictions[0], axis=1)[0]
        fertilization_pred_normalized = predictions[1][0][0]  # Normalized [0, 1]
        damage_pred = predictions[2][0][0] * 100  # Convert to percentage
        yield_pred_normalized = predictions[3][0][0]  # Normalized [0, 1]

        # Denormalize predictions
        fertilization_pred = fertilization_pred_normalized * (FERTILIZATION_MAX - FERTILIZATION_MIN) + FERTILIZATION_MIN
        yield_pred = yield_pred_normalized * (YIELD_MAX - YIELD_MIN) + YIELD_MIN

        # Log normalized and denormalized values for debugging
        logger.info(f"Normalized yield prediction: {yield_pred_normalized}")
        logger.info(f"Denormalized yield prediction: {yield_pred}")

        result = {
            'crop_type': crop_types[crop_type_pred],
            'fertilization_required': f"{fertilization_pred:.2f}",
            'destruction_percentage': f"{damage_pred:.2f}",
            'yield_prediction': f"{yield_pred:.2f}"
        }
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {'error': str(e)}