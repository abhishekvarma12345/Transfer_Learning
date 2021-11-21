import argparse
import os
import numpy
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf


STAGE = "basic model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    ## get data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    ## define layers
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(300),
        tf.keras.layers.LeakyReLU(), ## alternate way
        tf.keras.layers.Dense(100),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(10, activation="softmax")
    ]

    ## define model
    model = tf.keras.models.Sequential(LAYERS)

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    ## compiling model
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

    # training model
    model.fit(X_train, y_train, 
    epochs=10,
    validation_data=(X_valid, y_valid),
    verbose=2)

    # save the base model
    model_dir_path = os.path.join("artifacts", "models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "base_model.h5")

    model.save(model_file_path)

    logging.info(f"base model saved at {model_file_path}")
    logging.info(f"evaluation metrics {model.evaluate(X_test, y_test)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e