"""The time-conditioned U-Net denoiser of Stage 1 (cells 16-18), copied unchanged in Stages 2 and 3.

`create_denoiser((208, 384, 1), base_filters=16)` has 508,785 parameters; with a 2-channel
input (Stage 1's conditional denoiser) 508,929. The saved `.keras` files in `models/` contain
a Lambda layer, so they are reloaded with `create_denoiser(...)` + `load_weights(path)`.
"""

import tensorflow as tf


def sinusoidal_embedding(t, dim=64):
    """t: float tensor, shape (batch,). Returns shape (batch, dim)."""
    half = dim // 2
    freqs = tf.exp(-tf.math.log(10000.0) * tf.range(half, dtype=tf.float32) / half)
    args = tf.cast(t[:, None], tf.float32) * freqs[None, :]
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


def conv_block_t(x, t_emb, filters, kernel_size=3, activation="relu", padding="same"):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding,
                                kernel_initializer="he_normal")(x)

    # Project the time embedding to `filters` channels and add it (broadcast
    # over height/width) -- this is how the network learns "how noisy am I".
    t_proj = tf.keras.layers.Dense(filters)(t_emb)
    t_proj = tf.keras.layers.Reshape((1, 1, filters))(t_proj)
    x = tf.keras.layers.Add()([x, t_proj])
    x = tf.keras.layers.Activation(activation)(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation,
                                padding=padding, kernel_initializer="he_normal")(x)
    return x


def encoder_block_t(x, t_emb, filters):
    skip = conv_block_t(x, t_emb, filters)
    down = tf.keras.layers.MaxPooling2D(pool_size=2)(skip)
    return skip, down


def decoder_block_t(x, skip, t_emb, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block_t(x, t_emb, filters)
    return x


def create_denoiser(input_shape, base_filters=16, time_emb_dim=64):
    image_input = tf.keras.layers.Input(shape=input_shape, name="noisy_field")
    t_input = tf.keras.layers.Input(shape=(), name="timestep")

    t_emb = tf.keras.layers.Lambda(
        lambda t: sinusoidal_embedding(t, dim=time_emb_dim)
    )(t_input)
    t_emb = tf.keras.layers.Dense(time_emb_dim, activation="relu")(t_emb)

    skip1, x = encoder_block_t(image_input, t_emb, base_filters)
    skip2, x = encoder_block_t(x, t_emb, base_filters * 2)
    skip3, x = encoder_block_t(x, t_emb, base_filters * 4)

    x = conv_block_t(x, t_emb, base_filters * 8)

    x = decoder_block_t(x, skip3, t_emb, base_filters * 4)
    x = decoder_block_t(x, skip2, t_emb, base_filters * 2)
    x = decoder_block_t(x, skip1, t_emb, base_filters)

    noise_pred = tf.keras.layers.Conv2D(1, kernel_size=1, activation="linear")(x)

    return tf.keras.Model([image_input, t_input], noise_pred, name="Denoiser")


def load_denoiser(weights_path, input_shape=(208, 384, 1), base_filters=16):
    """Rebuild the architecture and load weights from a saved `.keras` file (README's recipe)."""
    denoiser = create_denoiser(input_shape, base_filters=base_filters)
    denoiser.load_weights(str(weights_path))
    return denoiser
