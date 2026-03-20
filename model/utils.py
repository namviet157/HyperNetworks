import tensorflow as tf


def _to_2tuple(value):
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError('kernel_size and strides must be an int or a pair of ints.')
    return tuple(value)


class HyperConv2D(tf.keras.layers.Layer):
    """Convolution layer whose kernel is produced by a static hypernetwork."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='same',
        use_bias=True,
        in_block_size=16,
        out_block_size=16,
        z_dim=4,
        kernel_initializer=None,
        bias_initializer='zeros',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = _to_2tuple(kernel_size)
        self.strides = _to_2tuple(strides)
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.in_block_size = int(in_block_size)
        self.out_block_size = int(out_block_size)
        self.z_dim = int(z_dim)
        self.kernel_initializer = kernel_initializer or tf.keras.initializers.RandomNormal(stddev=0.01)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        if input_channels % self.in_block_size != 0:
            raise ValueError(
                'Input channels (%d) must be divisible by in_block_size (%d).'
                % (input_channels, self.in_block_size)
            )
        if self.filters % self.out_block_size != 0:
            raise ValueError(
                'Output channels (%d) must be divisible by out_block_size (%d).'
                % (self.filters, self.out_block_size)
            )

        self.input_channels = input_channels
        self.num_in_blocks = input_channels // self.in_block_size
        self.num_out_blocks = self.filters // self.out_block_size

        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.num_in_blocks, self.num_out_blocks, self.z_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )
        self.w1 = self.add_weight(
            name='w1',
            shape=(self.z_dim, self.in_block_size * self.z_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.b1 = self.add_weight(
            name='b1',
            shape=(self.in_block_size * self.z_dim,),
            initializer='zeros',
            trainable=True,
        )
        self.w2 = self.add_weight(
            name='w2',
            shape=(self.z_dim, self.kernel_size[0] * self.kernel_size[1] * self.out_block_size),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(self.kernel_size[0] * self.kernel_size[1] * self.out_block_size,),
            initializer='zeros',
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def _generate_kernel(self):
        input_blocks = []
        for i in range(self.num_in_blocks):
            output_blocks = []
            for j in range(self.num_out_blocks):
                z = self.embeddings[i, j]
                z = tf.reshape(z, (1, self.z_dim))
                a = tf.matmul(z, self.w1) + self.b1
                a = tf.reshape(a, (self.in_block_size, self.z_dim))
                weight = tf.matmul(a, self.w2) + self.b2
                weight = tf.reshape(
                    weight,
                    (self.in_block_size, self.out_block_size, self.kernel_size[0], self.kernel_size[1]),
                )
                weight = tf.transpose(weight, (2, 3, 0, 1))
                output_blocks.append(weight)
            input_blocks.append(tf.concat(output_blocks, axis=3))
        return tf.concat(input_blocks, axis=2)

    def call(self, inputs):
        kernel = self._generate_kernel()
        outputs = tf.nn.conv2d(
            inputs,
            kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding,
        )
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding.lower(),
                'use_bias': self.use_bias,
                'in_block_size': self.in_block_size,
                'out_block_size': self.out_block_size,
                'z_dim': self.z_dim,
                'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
                'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config


def hyper_config(in_block_size, out_block_size, z_dim):
    return {
        'in_block_size': int(in_block_size),
        'out_block_size': int(out_block_size),
        'z_dim': int(z_dim),
    }
