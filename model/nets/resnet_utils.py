import tensorflow as tf

from model.utils import HyperConv2D


def _to_2tuple(value):
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError('Expected an int or a tuple of length 2.')
    return tuple(value)


def conv2d_same_padding(inputs, kernel_size, strides):
    kernel_size = _to_2tuple(kernel_size)
    strides = _to_2tuple(strides)
    if strides == (1, 1):
        return inputs

    pad_along_height = max(kernel_size[0] - strides[0], 0)
    pad_along_width = max(kernel_size[1] - strides[1], 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])


def make_conv_layer(
    filters,
    kernel_size,
    strides=1,
    padding='same',
    use_bias=False,
    hyper_params=None,
    name=None,
):
    if hyper_params is None:
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=name,
        )
    return HyperConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        in_block_size=hyper_params['in_block_size'],
        out_block_size=hyper_params['out_block_size'],
        z_dim=hyper_params['z_dim'],
        name=name,
    )
