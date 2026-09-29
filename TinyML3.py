import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# SETTINGS

RANDOM_SEED = 92

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Excel dataset path

DATASET_PATH = r"C:\Users\farah\Downloads\trineta.xlsx"

# If data is not in the first Excel sheet, change this:
# EXCEL_SHEET_NAME = "Sheet1"
# Otherwise keep None

EXCEL_SHEET_NAME = None

# SELECTED 7 FEATURES
# ESP32 code must use exactly this same feature order.

FEATURES = [
    "distance_m",
    "closing_speed_mps",
    "ttc_s"
]

LABEL_COLUMN = "label"
# GENERATED FILE NAMES

KERAS_MODEL_FILENAME = "trinetra_3feature_model.keras"

TFLITE_FILENAME = "trinetra_3feature_int8.tflite"

MODEL_HEADER_FILENAME = "trinetra_3feature_model_data.h"

PREPROCESSING_JSON_FILENAME = "trinetra_3feature_preprocessing.json"

PREPROCESSING_HEADER_FILENAME = "trinetra_3feature_preprocessing.h"

MODEL_ARRAY_NAME = "trinetra_3feature_model"

# LOAD EXCEL DATASET

print("==============================================")
print(" TRINETRA 7-FEATURE TINYML TRAINING")
print("==============================================")

if EXCEL_SHEET_NAME is None:
    df = pd.read_excel(DATASET_PATH)
else:
    df = pd.read_excel(
        DATASET_PATH,
        sheet_name=EXCEL_SHEET_NAME
    )

print("\nDataset shape:", df.shape)

print("\nDataset columns:")

print(df.columns.tolist())

# CHECK REQUIRED COLUMNS

required_columns = FEATURES + [LABEL_COLUMN]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nERROR: Required columns are missing:\n"
        f"{missing_columns}\n\n"
        "Check your Excel column names."
    )

# Keep only selected features and label.
# Remove rows containing empty values.

df = df[required_columns].dropna().copy()

print("\nUsable dataset shape:", df.shape)

print("\nClass distribution:")

print(df[LABEL_COLUMN].value_counts())

print("\nClass distribution percentage:")

print(
    (
        df[LABEL_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
)
# PREPARE INPUT FEATURES AND LABELS

X = df[FEATURES].astype(np.float32).values

y_text = df[LABEL_COLUMN].astype(str).values

encoder = LabelEncoder()

y = encoder.fit_transform(y_text)

NUM_FEATURES = len(FEATURES)

NUM_CLASSES = len(encoder.classes_)

print("\n==============================================")
print(" CLASS ID MAPPING")
print("==============================================")

for class_id, class_name in enumerate(encoder.classes_):
    print(f"{class_id} -> {class_name}")

if NUM_CLASSES != 3:
    print(
        "\nWARNING: The dataset does not contain exactly "
        "three output classes."
    )

# TRAIN / VALIDATION / TEST SPLIT

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y_train_full
)

print("\n==============================================")
print(" DATA SPLIT")
print("==============================================")

print("Training   :", X_train.shape)

print("Validation :", X_val.shape)

print("Testing    :", X_test.shape)

# NORMALIZE FEATURES
# StandardScaler:
# normalized = (raw_value - mean) / standard_deviation
# ESP32 must use the same mean/std values.
# These values will be automatically saved inside:
# trinetra_preprocessing.h

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
).astype(np.float32)

X_val_scaled = scaler.transform(
    X_val
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)

# CLASS WEIGHTS
# Helps if classes have unequal number of samples.

unique_classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        unique_classes,
        weights
    )
}

print("\nClass weights:")

print(class_weights)

# BUILD TINYML NEURAL NETWORK
# Input  : 7 features
# Hidden : 16 neurons
# Hidden : 8 neurons
# Output : 3 classes

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(NUM_FEATURES,),
        name="vehicle_features"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_1"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu",
        name="dense_2"
    ),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="output"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================================")
print(" 7-FEATURE TINYML MODEL")
print("==============================================")

model.summary()

# TRAIN MODEL

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1
    )
]

history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(
        X_val_scaled,
        y_val
    ),
    epochs=150,
    batch_size=16,
    verbose=1,
    callbacks=callbacks,
    class_weight=class_weights
)
# EVALUATE KERAS MODEL

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\n==============================================")
print(" KERAS MODEL TEST RESULTS")
print("==============================================")

print(f"Test loss     : {test_loss:.4f}")

print(f"Test accuracy : {test_accuracy:.4f}")

# CLASSIFICATION REPORT

probabilities = model.predict(
    X_test_scaled,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# SAVE KERAS MODEL

model.save(KERAS_MODEL_FILENAME)

print("\nSaved Keras model:")

print(KERAS_MODEL_FILENAME)

# SAVE PREPROCESSING JSON

preprocessing_data = {
    "features": FEATURES,
    "labels": encoder.classes_.tolist(),
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist()
}

with open(
    PREPROCESSING_JSON_FILENAME,
    "w"
) as f:
    json.dump(
        preprocessing_data,
        f,
        indent=4
    )

print("\nSaved preprocessing JSON:")

print(PREPROCESSING_JSON_FILENAME)

# CREATE ESP32 PREPROCESSING HEADE

with open(
    PREPROCESSING_HEADER_FILENAME,
    "w"
) as f:

    f.write("#ifndef TRINETRA_PREPROCESSING_H\n")
    f.write("#define TRINETRA_PREPROCESSING_H\n\n")

    f.write(
        f"#define TRINETRA_NUM_FEATURES {NUM_FEATURES}\n"
    )

    f.write(
        f"#define TRINETRA_NUM_CLASSES {NUM_CLASSES}\n\n"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    f.write(
        "const char* const trinetra_labels"
        "[TRINETRA_NUM_CLASSES] = {\n"
    )

    for i, label in enumerate(encoder.classes_):
        comma = "," if i < NUM_CLASSES - 1 else ""

        f.write(f'    "{label}"{comma}\n')

    f.write("};\n\n")

    # --------------------------------------------------------
    # Feature mean
    # --------------------------------------------------------

    f.write(
        "const float trinetra_feature_mean"
        "[TRINETRA_NUM_FEATURES] = {\n"
    )

    for i, value in enumerate(scaler.mean_):
        comma = "," if i < NUM_FEATURES - 1 else ""

        f.write(
            f"    {value:.10f}f"
            f"{comma}"
            f"  // {FEATURES[i]}\n"
        )

    f.write("};\n\n")

    # --------------------------------------------------------
    # Feature standard deviation
    # --------------------------------------------------------

    f.write(
        "const float trinetra_feature_std"
        "[TRINETRA_NUM_FEATURES] = {\n"
    )

    for i, value in enumerate(scaler.scale_):
        comma = "," if i < NUM_FEATURES - 1 else ""

        f.write(
            f"    {value:.10f}f"
            f"{comma}"
            f"  // {FEATURES[i]}\n"
        )

    f.write("};\n\n")

    f.write("#endif\n")

print("\nSaved ESP32 preprocessing header:")

print(PREPROCESSING_HEADER_FILENAME)

# REPRESENTATIVE DATASET FOR INT8 QUANTIZATION

def representative_dataset():

    calibration_count = min(
        300,
        len(X_train_scaled)
    )

    for i in range(calibration_count):

        sample = X_train_scaled[
            i:i + 1
        ].astype(np.float32)

        yield [sample]

# CONVERT TO INT8 TFLITE MODEL

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8

converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(
    TFLITE_FILENAME,
    "wb"
) as f:
    f.write(tflite_model)

print("\n==============================================")
print(" INT8 TFLITE MODEL CREATED")
print("==============================================")

print("Saved:", TFLITE_FILENAME)

print(
    "TFLite model size:",
    len(tflite_model),
    "bytes"
)

# VERIFY INT8 TFLITE MODEL

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_FILENAME
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)

print("\n==============================================")
print(" INT8 QUANTIZATION SETTINGS")
print("==============================================")

print("Input scale      :", input_scale)

print("Input zero point :", input_zero_point)

print("Output scale      :", output_scale)

print("Output zero point :", output_zero_point)

tflite_predictions = []

for sample in X_test_scaled:

    sample = sample.reshape(
        1,
        NUM_FEATURES
    ).astype(np.float32)

    quantized_input = np.round(
        sample / input_scale +
        input_zero_point
    )

    quantized_input = np.clip(
        quantized_input,
        -128,
        127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"],
        quantized_input
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    predicted_class = int(
        np.argmax(output)
    )

    tflite_predictions.append(
        predicted_class
    )

tflite_accuracy = accuracy_score(
    y_test,
    tflite_predictions
)

print("\n==============================================")
print(" INT8 TFLITE VERIFICATION")
print("==============================================")

print(
    f"TFLite INT8 accuracy: "
    f"{tflite_accuracy:.4f}"
)

# CREATE ESP32 MODEL HEADER

with open(
    TFLITE_FILENAME,
    "rb"
) as f:
    model_bytes = f.read()

with open(
    MODEL_HEADER_FILENAME,
    "w"
) as f:

    f.write("#ifndef TRINETRA_TINYML_MODEL_DATA_H\n")
    f.write("#define TRINETRA_TINYML_MODEL_DATA_H\n\n")

    f.write("#include <cstdint>\n\n")

    f.write(
        f"alignas(16) const unsigned char "
        f"{MODEL_ARRAY_NAME}[] = {{\n"
    )

    for i, byte in enumerate(model_bytes):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}")

        if i != len(model_bytes) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    if len(model_bytes) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")

    f.write(
        f"const unsigned int "
        f"{MODEL_ARRAY_NAME}_len = "
        f"{len(model_bytes)};\n\n"
    )

    f.write("#endif\n")

print("\nSaved ESP32 TinyML model header:")

print(MODEL_HEADER_FILENAME)

# FINAL OUTPUT

print("\n==============================================")
print(" TRINETRA 7-FEATURE TINYML EXPORT COMPLETE")
print("==============================================")

print("\nGenerated files:")

print("1.", KERAS_MODEL_FILENAME)

print("2.", TFLITE_FILENAME)

print("3.", MODEL_HEADER_FILENAME)

print("4.", PREPROCESSING_JSON_FILENAME)

print("5.", PREPROCESSING_HEADER_FILENAME)

print("\nCopy these two files to the ESP32 Arduino folder:")

print("-", MODEL_HEADER_FILENAME)

print("-", PREPROCESSING_HEADER_FILENAME)

print("\nNEW ESP32 INPUT QUANTIZATION VALUES:")

print("INPUT_SCALE      =", input_scale)

print("INPUT_ZERO_POINT =", input_zero_point)

print("\nNEW ESP32 OUTPUT QUANTIZATION VALUES:")

print("OUTPUT_SCALE      =", output_scale)

print("OUTPUT_ZERO_POINT =", output_zero_point)