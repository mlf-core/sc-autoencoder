import tensorflow as tf
import os
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def train(model, epochs, train_dataset):
    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    ]

    model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))


def test(model, eval_dataset, save_path="embedding.png"):

    # Get latent space encoding
    res = np.array(model.encode(eval_dataset))

    # Reduce with UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(res)
    df = pd.DataFrame(embedding, columns=['PC1', 'PC2'])
    plt.scatter(df['PC1'], df['PC2'])
    plt.savefig(save_path)

    return save_path
