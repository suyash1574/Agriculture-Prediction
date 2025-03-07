# src/models/train_model.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"GPUs detected: {len(gpus)}")
else:
    logger.warning("No GPUs detected, using CPU")

# Load data
logger.info("Loading data")
try:
    image_df = pd.read_csv("data/processed/image_labels.csv")
    tabular_df = pd.read_csv("data/processed/tabular.csv")
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    raise

# Validate columns
required_image_cols = ["path", "crop_type"]
required_tabular_cols = ["Rainfall"]
if any(col not in image_df.columns for col in required_image_cols):
    logger.error(f"Missing columns in image_df")
    raise ValueError("Required columns missing in image dataset")
if any(col not in tabular_df.columns for col in required_tabular_cols):
    logger.error(f"Missing columns in tabular_df")
    raise ValueError("Required columns missing in tabular dataset")

# Prepare tabular data with standardization
tabular_data = np.array([[tabular_df["Rainfall"].mean()]])
tabular_data = np.tile(tabular_data, (len(image_df), 1)).astype(np.float32)
scaler = StandardScaler()
tabular_data = scaler.fit_transform(tabular_data)

# Prepare labels
crop_labels = LabelEncoder().fit_transform(image_df["crop_type"])
if "fertilization_required" not in image_df.columns:
    logger.warning("Simulating fertilization_required")
    image_df["fertilization_required"] = np.random.uniform(0.0, 100.0, size=len(image_df))
fertilization_labels = image_df["fertilization_required"].values.astype(np.float32)
fertilization_labels = np.clip(fertilization_labels, 0.0, 100.0)
fert_min = np.min(fertilization_labels)
fert_max = np.max(fertilization_labels)
fertilization_labels = (fertilization_labels - fert_min) / (fert_max - fert_min)
if "damage" not in image_df.columns:
    logger.warning("Simulating damage")
    image_df["damage"] = np.random.uniform(0.1, 0.9, size=len(image_df))
damage_labels = image_df["damage"].values.astype(np.float32)
yield_labels = image_df["Yield"].values if "Yield" in image_df.columns else tabular_df["Yield"].values[:len(image_df)]
yield_min = np.nanmin(yield_labels)
yield_max = np.nanmax(yield_labels)
yield_labels = (yield_labels - yield_min) / (yield_max - yield_min)
yield_labels = np.nan_to_num(yield_labels, nan=0.0).astype(np.float32)

# Save scaling params
scaling_params = {'yield_min': yield_min, 'yield_max': yield_max, 'fert_min': fert_min, 'fert_max': fert_max}
with open('models/scaling_params.pkl', 'wb') as f:
    pickle.dump(scaling_params, f)
logger.info(f"Scaling params saved: {scaling_params}")

# Split data
image_paths = image_df["path"].values
(
    image_paths_train, image_paths_val,
    tabular_train, tabular_val,
    crop_labels_train, crop_labels_val,
    fertilization_labels_train, fertilization_labels_val,
    damage_labels_train, damage_labels_val,
    yield_labels_train, yield_labels_val
) = train_test_split(
    image_paths, tabular_data, crop_labels, fertilization_labels, damage_labels, yield_labels,
    test_size=0.2, random_state=42
)

# Create TensorFlow dataset
def load_and_preprocess_image(path):
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.random_flip_left_right(img)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img
    except tf.errors.InvalidArgumentError as e:
        logger.warning(f"Failed to decode image at {path}: {str(e)}")
        return tf.zeros([224, 224, 3], dtype=tf.float32)

train_image_dataset = tf.data.Dataset.from_tensor_slices(image_paths_train).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_tabular_dataset = tf.data.Dataset.from_tensor_slices(tabular_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices({
    "crop_type": crop_labels_train,
    "fertilization": fertilization_labels_train,
    "damage": damage_labels_train,
    "yield": yield_labels_train
})
train_inputs = tf.data.Dataset.zip((train_image_dataset, train_tabular_dataset))
train_dataset = tf.data.Dataset.zip((train_inputs, train_labels_dataset)).batch(8).prefetch(tf.data.AUTOTUNE)

val_image_dataset = tf.data.Dataset.from_tensor_slices(image_paths_val).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_tabular_dataset = tf.data.Dataset.from_tensor_slices(tabular_val)
val_labels_dataset = tf.data.Dataset.from_tensor_slices({
    "crop_type": crop_labels_val,
    "fertilization": fertilization_labels_val,
    "damage": damage_labels_val,
    "yield": yield_labels_val
})
val_inputs = tf.data.Dataset.zip((val_image_dataset, val_tabular_dataset))
val_dataset = tf.data.Dataset.zip((val_inputs, val_labels_dataset)).batch(8).prefetch(tf.data.AUTOTUNE)

# Build model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = Flatten()(base_model.output)
x = Dropout(0.8)(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
crop_output = Dense(len(np.unique(crop_labels)), activation="softmax", name="crop_type")(x)
fertilization_output = Dense(1, activation="linear", name="fertilization")(x)
damage_output = Dense(1, activation="sigmoid", name="damage")(x)
tabular_input = Input(shape=(1,), dtype=tf.float32)
y = Dense(32, activation="relu")(tabular_input)
yield_output = Dense(1, activation="sigmoid", name="yield")(y)

model = Model(inputs=[base_model.input, tabular_input], outputs=[crop_output, fertilization_output, damage_output, yield_output])
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss={"crop_type": "sparse_categorical_crossentropy", "fertilization": "mse", "damage": "mse", "yield": "mse"},
    loss_weights={"crop_type": 1.0, "fertilization": 0.01, "damage": 1.0, "yield": 1.0},
    metrics={"crop_type": "accuracy", "fertilization": "mae", "damage": "mae", "yield": "mae"}
)

logger.info("Model summary:")
model.summary(print_fn=lambda x: logger.info(x))

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
class PredictionLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(val_dataset, steps=1)
        logger.info(f"Epoch {epoch} - Sample predictions: {predictions}")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping, reduce_lr, PredictionLogger()]
)

logger.info("Training metrics:")
for metric_name, metric_values in history.history.items():
    logger.info(f"{metric_name}: {metric_values[-1]}")

os.makedirs("models", exist_ok=True)
model.save("models/crop_model.h5")
logger.info("Model saved to models/crop_model.h5")