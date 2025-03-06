import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
image_df = pd.read_csv("data/processed/image_labels.csv")
tabular_df = pd.read_csv("data/processed/tabular.csv")

# Debug: Print number of images and tabular rows
print(f"Number of images: {len(image_df)}")
print(f"Number of tabular rows: {len(tabular_df)}")

# Prepare labels
crop_labels = LabelEncoder().fit_transform(image_df["crop_type"])
damage_labels = image_df["damage"].values

# Ensure tabular data aligns with image data
num_samples = len(image_df)
if len(tabular_df) < num_samples:
    tabular_df = tabular_df.iloc[:num_samples].reindex(range(num_samples)).fillna(0)
else:
    tabular_df = tabular_df.iloc[:num_samples]

# Prepare labels with normalization
yield_labels = tabular_df["Yield"].values
# Normalize yield_labels to [0, 1] based on min/max
yield_min, yield_max = yield_labels.min(), yield_labels.max()
yield_labels = (yield_labels - yield_min) / (yield_max - yield_min) if yield_max > yield_min else yield_labels

fert_labels = pd.get_dummies(np.random.randint(0, 3, num_samples)).values  # Placeholder

# Debug: Check label ranges
print(f"crop_labels min/max: {crop_labels.min()}/{crop_labels.max()}")
print(f"damage_labels min/max: {damage_labels.min()}/{damage_labels.max()}")
print(f"yield_labels min/max: {yield_labels.min()}/{yield_labels.max()} (normalized)")
print(f"fert_labels shape: {fert_labels.shape}, min/max: {fert_labels.min()}/{fert_labels.max()}")

# Clip and cast labels to avoid overflow
crop_labels = np.clip(crop_labels, 0, 1000).astype(np.int32)  # Ensure reasonable range
damage_labels = np.clip(damage_labels, 0, 1).astype(np.float32)  # Damage should be [0, 1]
yield_labels = yield_labels.astype(np.float32)  # Already normalized
fert_labels = fert_labels.astype(np.float32)

# Tabular data: dynamically select available columns
tabular_columns = ["Rainfall"]
if "Temperature" in tabular_df.columns:
    tabular_columns.append("Temperature")
tabular_data = tabular_df[tabular_columns].values.astype(np.float32)  # Ensure float32

# Split the data into training and validation sets
val_split = 0.2
image_paths = image_df["path"].values
(
    image_paths_train, image_paths_val,
    tabular_train, tabular_val,
    crop_labels_train, crop_labels_val,
    damage_labels_train, damage_labels_val,
    yield_labels_train, yield_labels_val,
    fert_labels_train, fert_labels_val
) = train_test_split(
    image_paths,
    tabular_data,
    crop_labels,
    damage_labels,
    yield_labels,
    fert_labels,
    test_size=val_split,
    random_state=42,
    shuffle=True
)

# Debug: Print split sizes
print(f"Training samples: {len(image_paths_train)}")
print(f"Validation samples: {len(image_paths_val)}")

# Create a TensorFlow dataset for images
def load_and_preprocess_image(path):
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)  # Force 3 channels
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
        img = tf.image.resize(img, [224, 224])  # Resize
        return img
    except tf.errors.InvalidArgumentError as e:
        logger.warning(f"Failed to decode image at {path}: {str(e)}")
        return tf.zeros([224, 224, 3], dtype=tf.float32)  # Return a zero image as placeholder

# Create training dataset
train_image_dataset = tf.data.Dataset.from_tensor_slices(image_paths_train)
train_image_dataset = train_image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_tabular_dataset = tf.data.Dataset.from_tensor_slices(tabular_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices({
    "crop_type": crop_labels_train,
    "damage": damage_labels_train,
    "yield": yield_labels_train,
    "fertilization": fert_labels_train
})
train_dataset = tf.data.Dataset.zip((train_image_dataset, train_tabular_dataset), train_labels_dataset)
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Create validation dataset
val_image_dataset = tf.data.Dataset.from_tensor_slices(image_paths_val)
val_image_dataset = val_image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_tabular_dataset = tf.data.Dataset.from_tensor_slices(tabular_val)
val_labels_dataset = tf.data.Dataset.from_tensor_slices({
    "crop_type": crop_labels_val,
    "damage": damage_labels_val,
    "yield": yield_labels_val,
    "fertilization": fert_labels_val
})
val_dataset = tf.data.Dataset.zip((val_image_dataset, val_tabular_dataset), val_labels_dataset)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Debug: Check dataset sizes
train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
print(f"Train batches: {train_batches}, samples: {train_batches * 32}")
print(f"Validation batches: {val_batches}, samples: {val_batches * 32}")

# VGG16 base
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Image branch
x = Flatten()(base_model.output)
crop_output = Dense(len(np.unique(crop_labels)), activation="softmax", name="crop_type")(x)
damage_output = Dense(1, activation="sigmoid", name="damage")(x)

# Tabular branch
tabular_input = Input(shape=(len(tabular_columns),))
y = Dense(64, activation="relu")(tabular_input)
yield_output = Dense(1, activation="linear", name="yield")(y)
fert_output = Dense(3, activation="softmax", name="fertilization")(y)

# Model
model = Model(inputs=[base_model.input, tabular_input], outputs=[crop_output, damage_output, yield_output, fert_output])
model.compile(
    optimizer="adam",
    loss={
        "crop_type": "sparse_categorical_crossentropy",
        "damage": "mse",
        "yield": "mse",
        "fertilization": "categorical_crossentropy"
    },
    loss_weights={
        "crop_type": 1.0,
        "damage": 1.0,
        "yield": 0.01,  # Reduce weight to balance with other losses
        "fertilization": 1.0
    },
    metrics={
        "crop_type": "accuracy",
        "damage": "mae",
        "yield": "mae",
        "fertilization": "accuracy"
    }
)

# Train
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/crop_model.h5")
print("Saved model to models/crop_model.h5")