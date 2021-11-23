import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import io


STAGE = "transfer model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def even_odd_label(y_train, y_test, y_valid):
    y_train_bin = np.where(y_train % 2 == 0, 1, 0)
    y_test_bin = np.where(y_test % 2 == 0, 1, 0)
    y_valid_bin = np.where(y_valid % 2 == 0, 1, 0)
    return (y_train_bin, y_test_bin, y_valid_bin)

def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    ## get data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    y_train_bin, y_test_bin, y_valid_bin = even_odd_label(y_train, y_test, y_valid)
    ## set the seeds
    seed = 2021 ## get it from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model_file_path = os.path.join("artifacts","models","base_model.h5")
    ## load base model
    base_model = tf.keras.models.load_model(model_file_path)

    # log our model summary info in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_string = stream.getvalue()
        return summary_string

    ## model summary
    logging.info(f"base model summary: \n{_log_model_summary(base_model)}")

    base_model = base_model.layers[:-1]
    ## define transfer model
    for layer in base_model:
        layer.trainable = False
    
    base_model.append(tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer"))
    
    transfer_model = tf.keras.models.Sequential(base_model)

    LOSS = "binary_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    ## compiling model
    transfer_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)


    # log our model summary info in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_string = stream.getvalue()
        return summary_string

    ## model summary
    logging.info(f"{STAGE} summary: \n{_log_model_summary(transfer_model)}")

    # training model
    transfer_model.fit(X_train, y_train_bin, 
    epochs=10,
    validation_data=(X_valid, y_valid_bin),
    verbose=2)

    # save the base model
    model_dir_path = os.path.join("artifacts", "models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "transfer_model.h5")

    transfer_model.save(model_file_path)

    logging.info(f"{STAGE} saved at {model_file_path}")
    logging.info(f"{STAGE} evaluation metrics {transfer_model.evaluate(X_test, y_test_bin)}")

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