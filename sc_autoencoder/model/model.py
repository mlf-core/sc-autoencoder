import tensorflow as tf


class Autoencoder(tf.keras.Model):

    def __init__(self, input_dim, layer_sizes, latent_size):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.latent_size = latent_size

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.layer_sizes[0],  activation="relu", name="encoder1"),
            tf.keras.layers.Dense(self.layer_sizes[1],  activation="relu", name="encoder2"),
            tf.keras.layers.Dense(self.layer_sizes[2],  activation="relu", name="encoder3"),
            tf.keras.layers.Dense(self.latent_size,     activation="relu", name="latent")
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.layer_sizes[2],  activation="relu", name="decoder1"),
            tf.keras.layers.Dense(self.layer_sizes[1],  activation="relu", name="decoder2"),
            tf.keras.layers.Dense(self.layer_sizes[0],  activation="relu", name="decoder3"),
            tf.keras.layers.Dense(self.input_dim,     activation="linear", name="out_layer")
        ])

    
    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def encode(self, x):
        return self.encoder(x)


def create_model(input_shape):
    model = Autoencoder(input_dim=input_shape, layer_sizes=[256, 128, 64], latent_size=50)
    return model
