import tensorflow as tf


def _max_kernel_spatial(kernel_size):
    kh, kw = kernel_size
    return max(int(kh), int(kw))


def _to_2tuple(value):
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError('kernel_size and strides must be an int or a pair of ints.')
    return tuple(value)


class SharedHyperConvMLP(tf.keras.layers.Layer):
    """One hypernetwork (two-layer MLP) shared by many HyperConv2D layers.

    Output is always shaped for a k×k spatial patch with k = max_kernel_spatial so that
    1×1 convolutions can reuse the same weights by taking the center spatial slice.
    """

    def __init__(
        self,
        in_block_size,
        out_block_size,
        z_dim,
        max_kernel_spatial=3,
        kernel_initializer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_block_size = int(in_block_size)
        self.out_block_size = int(out_block_size)
        self.z_dim = int(z_dim)
        self.max_kernel_spatial = int(max_kernel_spatial)
        self.kernel_initializer = kernel_initializer or tf.keras.initializers.RandomNormal(stddev=0.01)

    def build(self, input_shape=None):
        k = self.max_kernel_spatial
        spatial_out = k * k * self.out_block_size
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
            shape=(self.z_dim, spatial_out),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(spatial_out,),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, z_row):
        """z_row: (1, z_dim) — single block embedding as a row vector."""
        a = tf.matmul(z_row, self.w1) + self.b1
        a = tf.nn.relu(a)
        a = tf.reshape(a, (self.in_block_size, self.z_dim))
        weight = tf.matmul(a, self.w2) + self.b2
        k = self.max_kernel_spatial
        weight = tf.reshape(
            weight,
            (self.in_block_size, self.out_block_size, k, k),
        )
        return weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'in_block_size': self.in_block_size,
                'out_block_size': self.out_block_size,
                'z_dim': self.z_dim,
                'max_kernel_spatial': self.max_kernel_spatial,
                'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config


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
        layer_embedding=False,
        shared_hypernet=None,
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
        self.layer_embedding = layer_embedding
        self.shared_hypernet = shared_hypernet
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
        if self.layer_embedding:
            self.z_layer = self.add_weight(
                name='z_layer',
                shape=(self.z_dim,),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                trainable=True,
            )
        else:
            self.z_layer = None

        if self.shared_hypernet is not None:
            if (
                self.shared_hypernet.in_block_size != self.in_block_size
                or self.shared_hypernet.out_block_size != self.out_block_size
                or self.shared_hypernet.z_dim != self.z_dim
            ):
                raise ValueError(
                    'shared_hypernet block sizes and z_dim must match HyperConv2D '
                    '(got shared %s vs layer in_block=%d out_block=%d z_dim=%d).'
                    % (
                        (self.shared_hypernet.in_block_size, self.shared_hypernet.out_block_size, self.shared_hypernet.z_dim),
                        self.in_block_size,
                        self.out_block_size,
                        self.z_dim,
                    )
                )
            k_layer = _max_kernel_spatial(self.kernel_size)
            if k_layer > self.shared_hypernet.max_kernel_spatial:
                raise ValueError(
                    'Kernel spatial size %s exceeds shared hypernet max_kernel_spatial=%d.'
                    % (self.kernel_size, self.shared_hypernet.max_kernel_spatial)
                )
            if not self.shared_hypernet.built:
                self.shared_hypernet.build(None)
            self.w1 = self.b1 = self.w2 = self.b2 = None
        else:
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

    def _hyper_forward(self, z_row):
        if self.shared_hypernet is not None:
            return self.shared_hypernet(z_row)
        a = tf.matmul(z_row, self.w1) + self.b1
        a = tf.nn.relu(a)
        a = tf.reshape(a, (self.in_block_size, self.z_dim))
        weight = tf.matmul(a, self.w2) + self.b2
        k_h, k_w = self.kernel_size
        weight = tf.reshape(
            weight,
            (self.in_block_size, self.out_block_size, k_h, k_w),
        )
        return weight

    def _crop_spatial_to_kernel(self, weight):
        """weight: (in_b, out_b, K, K) with K = shared max spatial; crop to self.kernel_size."""
        k_h, k_w = self.kernel_size
        k_max = weight.shape[2]
        k_max = int(k_max) if k_max is not None else self.shared_hypernet.max_kernel_spatial
        off_h = (k_max - k_h) // 2
        off_w = (k_max - k_w) // 2
        weight = weight[:, :, off_h : off_h + k_h, off_w : off_w + k_w]
        return weight

    def _generate_kernel(self):
        input_blocks = []
        for i in range(self.num_in_blocks):
            output_blocks = []
            for j in range(self.num_out_blocks):
                z = self.embeddings[i, j]
                if self.z_layer is not None:
                    z = z + self.z_layer
                z = tf.reshape(z, (1, self.z_dim))
                weight = self._hyper_forward(z)
                if self.shared_hypernet is not None:
                    weight = self._crop_spatial_to_kernel(weight)
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
                'layer_embedding': self.layer_embedding,
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