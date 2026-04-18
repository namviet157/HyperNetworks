import tensorflow as tf

from model.nets import resnet_utils
from model.utils import SharedHyperConvMLP, hyper_config


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, depth, bottleneck_depth, stride=1, hyper_params=None, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.bottleneck_depth = bottleneck_depth
        self.stride = stride
        self.hyper_params = hyper_params

        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)

        self.conv1 = resnet_utils.make_conv_layer(
            filters=bottleneck_depth,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            hyper_params=hyper_params,
            name='conv1',
        )
        self.conv2 = resnet_utils.make_conv_layer(
            filters=bottleneck_depth,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False,
            hyper_params=hyper_params,
            name='conv2',
        )
        self.conv3 = resnet_utils.make_conv_layer(
            filters=depth,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            hyper_params=hyper_params,
            name='conv3',
        )
        self.shortcut_pool = None
        self.shortcut_conv = None

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        if self.stride != 1 and input_channels == self.depth:
            self.shortcut_pool = tf.keras.layers.AveragePooling2D(pool_size=1, strides=self.stride)
        elif input_channels != self.depth:
            self.shortcut_conv = resnet_utils.make_conv_layer(
                filters=self.depth,
                kernel_size=1,
                strides=self.stride,
                padding='same',
                use_bias=False,
                hyper_params=self.hyper_params,
                name='shortcut',
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(x)
        elif self.shortcut_pool is not None:
            shortcut = self.shortcut_pool(inputs)
        else:
            shortcut = inputs

        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        return shortcut + x


class ResNetV2(tf.keras.Model):
    def __init__(self, layer_blocks, num_classes, hyper_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hyper_mode = hyper_mode
        if hyper_mode:
            self.shared_hyper_conv = SharedHyperConvMLP(
                64,
                64,
                64,
                max_kernel_spatial=3,
                name='shared_hyper_conv',
            )
            shared_hyper_params = {
                **hyper_config(64, 64, 64),
                'shared_hypernet': self.shared_hyper_conv,
                'layer_embedding': True,
            }
        else:
            self.shared_hyper_conv = None
            shared_hyper_params = None

        self.stem_conv = tf.keras.layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='stem_conv',
        )
        self.stem_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='stem_pool')

        self.stages = []
        stage_defs = [
            (64, layer_blocks[0], 1),
            (128, layer_blocks[1], 2),
            (256, layer_blocks[2], 2),
            (512, layer_blocks[3], 2),
        ]
        for stage_index, (base_depth, num_units, first_stride) in enumerate(stage_defs, start=1):
            blocks = []
            for unit_index in range(num_units):
                stride = first_stride if unit_index == 0 else 1
                blocks.append(
                    BottleneckBlock(
                        depth=base_depth * 4,
                        bottleneck_depth=base_depth,
                        stride=stride,
                        hyper_params=shared_hyper_params,
                        name='block%d_unit%d' % (stage_index, unit_index + 1),
                    )
                )
            self.stages.append(blocks)

        self.post_bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='post_bn')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')

    def call(self, inputs, training=False, return_endpoints=False):
        endpoints = {}
        x = self.stem_conv(inputs)
        x = self.stem_pool(x)
        endpoints['stem'] = x

        for stage_index, blocks in enumerate(self.stages, start=1):
            for block in blocks:
                x = block(x, training=training)
            endpoints['block%d' % stage_index] = x

        x = self.post_bn(x, training=training)
        x = tf.nn.relu(x)
        endpoints['postnorm'] = x
        x = self.global_pool(x)
        logits = self.classifier(x)
        endpoints['logits'] = logits
        endpoints['predictions'] = tf.nn.softmax(logits)
        if return_endpoints:
            return logits, endpoints
        return logits


def build_resnet_v2_50(num_classes, hyper_mode=False, name='resnet_v2_50'):
    return ResNetV2([3, 4, 6, 3], num_classes=num_classes, hyper_mode=hyper_mode, name=name)
