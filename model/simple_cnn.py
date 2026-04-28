import tensorflow as tf

from model.utils import HyperConv2D


class SimpleCNN(tf.keras.Model):
    def __init__(
        self,
        num_classes=10,
        f_size=7,
        in_size=16,
        out_size=16,
        batch_size=64,
        hyper_mode=True,
        conv_weight_initializer=None,
        kernel_initializer=None,
        bias_initializer=None,
    ):
        super().__init__(name='simple_cnn')
        del conv_weight_initializer

        self.num_classes = num_classes
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.hyper_mode = hyper_mode
        self.kernel_initializer = kernel_initializer or tf.keras.initializers.Orthogonal(gain=1.0)
        self.bias_initializer = bias_initializer or tf.keras.initializers.Constant(0.0)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=out_size,
            kernel_size=f_size,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer='zeros',
            name='conv1',
        )
        if hyper_mode:
            self.conv2 = HyperConv2D(
                filters=out_size,
                kernel_size=f_size,
                strides=1,
                padding='same',
                use_bias=True,
                in_block_size=in_size,
                out_block_size=out_size,
                z_dim=4,
                name='hyper_conv2',
            )
        else:
            self.conv2 = tf.keras.layers.Conv2D(
                filters=out_size,
                kernel_size=f_size,
                padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer='zeros',
                name='conv2',
            )
        # self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name='pool1')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name='pool2')
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(
            units=num_classes,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='classifier',
        )

    def call(self, inputs, training=False):
        del training
        x = self.conv1(inputs)
        # x = tf.nn.relu(x)
        x = self.relu1(x)
        # x = self.pool(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = tf.nn.relu(x)
        x = self.relu2(x)
        # x = self.pool(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return self.classifier(x)

    def build_graph(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
