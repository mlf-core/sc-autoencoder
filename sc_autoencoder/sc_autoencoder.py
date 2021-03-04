from argparse import ArgumentParser
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import time
from rich import traceback, print

from mlf_core.mlf_core import MLFCore
from rich import traceback
from mlf_core.mlf_core import log_sys_intel_conda_env, set_general_random_seeds
from data_loading.data_loader import load_data
from model.model import create_model
from training.train import train, test
from tensorflow.keras.mixed_precision import experimental as mp


def start_training():
    parser = ArgumentParser(description='Tensorflow example')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=True,
        help='Enable or disable CUDA support',
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=10,
        help='Number of epochs to train',
    )
    parser.add_argument(
        '--general-seed',
        type=int,
        default=0,
        help='General Python, Python random and Numpy seed.',
    )
    parser.add_argument(
        '--tensorflow-seed',
        type=int,
        default=0,
        help='Tensorflow specific random seed.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Input batch size for training and testing',
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=10000,
        help='Buffer size for Mirrored Training',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate',
    )
    args = parser.parse_args()
    dict_args = vars(args)

    # Disable GPU support if no GPUs are supposed to be used
    if not dict_args['cuda']:
        tf.config.set_visible_devices([], 'GPU')

    # Enable mixed precision training
    if mixed_precision:
        policy = mp.Policy('mixed_float16')
        mp.set_policy(policy)

    with mlflow.start_run():
        # Enable the logging of all parameters, metrics and models to mlflow and Tensorboard
        mlflow.autolog(1)

        # Log hardware and software
        MLFCore.log_sys_intel_conda_env()

        # Fix all random seeds and Tensorflow specific reproducibility settings
        MLFCore.set_general_random_seeds(dict_args["general_seed"])
        MLFCore.set_tensorflow_random_seeds(dict_args["tensorflow_seed"])

        # Use Mirrored Strategy for multi GPU support
        strategy = tf.distribute.MirroredStrategy()
        print(f'[bold blue]Number of devices: {strategy.num_replicas_in_sync}')

        # Fetch and prepare dataset

        dataset, test_data = load_data(strategy, dict_args['batch_size'], dict_args['buffer_size'], dict_args['tensorflow_seed'])

        # Get the input dimension
        input_dim = 0
        for elem in dataset:
            input_dim = elem[0].shape[1]
            break

        with strategy.scope():

            # Define and compile model
            model = create_model(input_shape=input_dim)
            model.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=dict_args['lr']),
                          metrics=['mse'])

            model.build(input_shape=(batch_size, input_dim))
            # Train and evaluate the trained model
            runtime = time.time()

            train(model, epochs, dataset)
            embedding = test(model, test_data, save_path="embedding.png")
            mlflow.log_artifact(embedding + ".png")
            mlflow.log_artifact(embedding + ".csv")

            device = 'GPU' if cuda else 'CPU'
            click.echo(click.style(f'{device} Run Time: {str(time.time() - runtime)} seconds', fg='green'))

            # Log hardware and software
            log_sys_intel_conda_env()

            click.echo(click.style(f'\nLaunch TensorBoard with:\ntensorboard --logdir={os.path.join(mlflow.get_artifact_uri(), "tensorboard_logs", "train")}',
                                   fg='blue'))

            device = 'GPU' if dict_args['cuda'] else 'CPU'
            print(f'[bold green]{device} Run Time: {str(time.time() - runtime)} seconds')

            print(f'[bold blue]\nLaunch TensorBoard with:\ntensorboard --logdir={os.path.join(mlflow.get_artifact_uri(), "tensorboard_logs", "train")}')



if __name__ == '__main__':
    traceback.install()
    print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Filtering out any Warnings messages

    start_training()
