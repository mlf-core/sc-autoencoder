import click
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import time

from rich import traceback

from mlf_core.mlf_core import log_sys_intel_conda_env, set_general_random_seeds
from data_loading.data_loader import load_data
from model.model import create_model
from training.train import train, test


@click.command()
@click.option('--cuda', type=bool, default=True, help='Enable or disable CUDA support')
@click.option('--epochs', type=int, default=2, help='Number of epochs to train')
@click.option('--general-seed', type=int, default=0, help='General Python, Python random and Numpy seed.')
@click.option('--tensorflow-seed', type=int, default=0, help='Tensorflow specific random seed.')
@click.option('--batch-size', type=int, default=64, help='Input batch size for training and testing')
@click.option('--buffer-size', type=int, default=10000, help='Buffer size for Mirrored Training')
@click.option('--learning-rate', type=float, default=0.01, help='Learning rate')
def start_training(cuda, epochs, general_seed, tensorflow_seed, batch_size, buffer_size, learning_rate):
    # Disable GPU support if no GPUs are supposed to be used
    if not cuda:
        tf.config.set_visible_devices([], 'GPU')

    with mlflow.start_run():
        # Enable the logging of all parameters, metrics and models to mlflow and Tensorboard
        mlflow.tensorflow.autolog()

        # Fix all random seeds and Tensorflow specific reproducibility settings
        set_general_random_seeds(general_seed)
        set_tensorflow_random_seeds(tensorflow_seed)

        # Use Mirrored Strategy for multi GPU support
        strategy = tf.distribute.MirroredStrategy()
        click.echo(click.style(f'Number of devices: {strategy.num_replicas_in_sync}', fg='blue'))

        # Fetch and prepare dataset
        dataset, test_data = load_data(strategy, batch_size, buffer_size, tensorflow_seed)

        # Get the input dimension
        # TODO: find a nicer, less ugly way of doing this
        input_dim = 0
        for elem in dataset:
            input_dim = elem[0].shape[1]

        with strategy.scope():
            # Define model and compile model
            model = create_model(input_shape=input_dim)
            model.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          metrics=['mse'])

            model.build(input_shape=(batch_size, input_dim))
            #mlflow.tensorflow.save_model(tf_saved_model_dir="model")
            # Train and evaluate the trained model
            runtime = time.time()
            train(model, epochs, dataset)
            embedding = test(model, test_data, save_path="embedding.png")
            mlflow.log_artifact(embedding)

            device = 'GPU' if cuda else 'CPU'
            click.echo(click.style(f'{device} Run Time: {str(time.time() - runtime)} seconds', fg='green'))

            # Log hardware and software
            log_sys_intel_conda_env()

            click.echo(click.style(f'\nLaunch TensorBoard with:\ntensorboard --logdir={os.path.join(mlflow.get_artifact_uri(), "tensorboard_logs", "train")}',
                                   fg='blue'))


def set_tensorflow_random_seeds(seed):
    tf.random.set_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads = 1  # CPU only
    tf.config.threading.set_inter_op_parallelism_threads = 1  # CPU only
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'


if __name__ == '__main__':
    traceback.install()
    print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Filtering out any Warnings messages

    start_training()
