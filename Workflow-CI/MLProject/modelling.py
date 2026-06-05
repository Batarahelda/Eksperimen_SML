import tensorflow as tf
import mlflow
import mlflow.keras

# ======================
# DATASET
# ======================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "plantvillage_preprocessing",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "plantvillage_preprocessing",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ======================
# AUGMENTATION
# ======================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ======================
# BASE MODEL
# ======================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ======================
# MODEL
# ======================

inputs = tf.keras.Input(shape=(160,160,3))

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ======================
# COMPILE
# ======================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# CALLBACK
# ======================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ======================
# MLFLOW
# ======================

mlflow.set_experiment("PlantVillage")

with mlflow.start_run():

    mlflow.log_param(
        "model",
        "MobileNetV2"
    )

    mlflow.log_param(
        "image_size",
        160
    )

    mlflow.log_param(
        "batch_size",
        32
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )

    mlflow.log_metric(
        "best_train_accuracy",
        max(history.history["accuracy"])
    )

    mlflow.log_metric(
        "best_val_accuracy",
        max(history.history["val_accuracy"])
    )

    mlflow.keras.log_model(
        model,
        "model"
    )