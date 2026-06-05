import tensorflow as tf
import mlflow
import itertools

IMG_SIZE = (160,160)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "plantvillage_preprocessing",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "plantvillage_preprocessing",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=32
)

mlflow.set_experiment("PlantVillage_Tuning")

learning_rates = [0.001, 0.0001]
dropouts = [0.2, 0.3]

for lr, dropout in itertools.product(
    learning_rates,
    dropouts
):

    with mlflow.start_run():

        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(160,160,3)
        )

        base_model.trainable = False

        inputs = tf.keras.Input(
            shape=(160,160,3)
        )

        x = tf.keras.applications.mobilenet_v2.preprocess_input(
            inputs
        )

        x = base_model(
            x,
            training=False
        )

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dropout(
            dropout
        )(x)

        outputs = tf.keras.layers.Dense(
            len(train_ds.class_names),
            activation="softmax"
        )(x)

        model = tf.keras.Model(
            inputs,
            outputs
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5,
            verbose=1
        )

        # ==========================
        # MANUAL LOGGING
        # ==========================

        mlflow.log_param(
            "learning_rate",
            lr
        )

        mlflow.log_param(
            "dropout",
            dropout
        )

        mlflow.log_param(
            "epochs",
            5
        )

        mlflow.log_metric(
            "train_accuracy",
            max(
                history.history["accuracy"]
            )
        )

        mlflow.log_metric(
            "val_accuracy",
            max(
                history.history["val_accuracy"]
            )
        )

        mlflow.log_metric(
            "train_loss",
            min(
                history.history["loss"]
            )
        )

        mlflow.log_metric(
            "val_loss",
            min(
                history.history["val_loss"]
            )
        )

        model.save(
            "best_model.keras"
        )

        mlflow.log_artifact(
            "best_model.keras"
        )